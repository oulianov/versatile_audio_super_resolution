import gc
import os
import re

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml

import audiosr.latent_diffusion.modules.phoneme_encoder.text as text
from audiosr.latent_diffusion.models.ddpm import LatentDiffusion
from audiosr.latent_diffusion.util import get_vits_phoneme_ids_no_padding
from audiosr.utils import (
    default_audioldm_config,
    download_checkpoint,
    lowpass_filtering_prepare_inference,
    normalize_wav,
    pad_wav,
    read_audio_file,
    wav_feature_extraction,
)


def seed_everything(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def text2phoneme(data):
    return text._clean_text(re.sub(r"<.*?>", "", data), ["english_cleaners2"])


def text_to_filename(text):
    return text.replace(" ", "_").replace("'", "_").replace('"', "_")


def extract_kaldi_fbank_feature(waveform, sampling_rate, log_mel_spec):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    if sampling_rate != 16000:
        waveform_16k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
    else:
        waveform_16k = waveform

    waveform_16k = waveform_16k - waveform_16k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    TARGET_LEN = log_mel_spec.size(0)

    # cut and pad
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LEN, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)

    return {"ta_kaldi_fbank": fbank}  # [1024, 128]


def make_batch_for_super_resolution(input_file, waveform=None, fbank=None):
    if waveform is None:
        # Original logic if no waveform is provided
        log_mel_spec, stft, waveform, duration, target_frame = read_audio_file(
            input_file
        )
    else:
        # New logic for chunk-based processing
        # We need to replicate the feature extraction from read_audio_file/wav_feature_extraction
        sampling_rate = 48000  # Assuming this is fixed
        duration = waveform.shape[-1] / sampling_rate

        # The original code pads to a multiple of 5.12s. We should do the same for each chunk
        # to match the model's expected input size.
        if duration % 5.12 != 0:
            pad_duration = duration + (5.12 - duration % 5.12)
        else:
            pad_duration = duration

        target_frame = int(pad_duration * 100)

        # Normalize and pad the waveform chunk
        waveform = normalize_wav(waveform)
        waveform = pad_wav(waveform, target_length=int(sampling_rate * pad_duration))

        log_mel_spec, stft = wav_feature_extraction(
            torch.from_numpy(waveform), target_frame
        )

    batch = {
        "waveform": torch.FloatTensor(waveform),
        "stft": torch.FloatTensor(stft),
        "log_mel_spec": torch.FloatTensor(log_mel_spec),
        "sampling_rate": 48000,
    }

    # print(batch["waveform"].size(), batch["stft"].size(), batch["log_mel_spec"].size())

    batch.update(lowpass_filtering_prepare_inference(batch))

    assert "waveform_lowpass" in batch.keys()
    lowpass_mel, lowpass_stft = wav_feature_extraction(
        batch["waveform_lowpass"], target_frame
    )
    batch["lowpass_mel"] = lowpass_mel

    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = torch.FloatTensor(batch[k]).unsqueeze(0)

    return batch, duration


def round_up_duration(duration):
    return int(round(duration / 2.5) + 1) * 2.5


def build_model(
    ckpt_path=None,
    config=None,
    device=None,
    model_name="basic",
    use_safetensors=False,
    compile=False,
    warmup_duration=20.48,
):
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print("Loading AudioSR: %s" % model_name)
    print("Loading model on %s" % device)

    if ckpt_path is None:
        ckpt_path = download_checkpoint(model_name, use_safetensors=use_safetensors)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)

    config["model"]["params"]["device"] = device
    config["model"]["params"]["use_ema"] = False

    # Initialize on meta device
    print("Initializing model structure...")
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    # Load weights first
    print(f"Loading weights from {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path, device="cpu")
        latent_diffusion.load_state_dict(state_dict, strict=False)
    else:
        checkpoint_data = torch.load(ckpt_path, map_location="cpu", mmap=True)
        state_dict = checkpoint_data["state_dict"]
        latent_diffusion.load_state_dict(state_dict, strict=False)

    # Move to target device
    latent_diffusion = latent_diffusion.to(device)
    latent_diffusion.eval()

    # Disable checkpointing for inference
    print("Disabling gradient checkpointing...")
    for module in latent_diffusion.modules():
        if hasattr(module, "checkpoint"):
            setattr(module, "checkpoint", False)
        if hasattr(module, "use_checkpoint"):
            setattr(module, "use_checkpoint", False)

    # Remove weight norm from vocoder for faster inference
    if hasattr(latent_diffusion, "first_stage_model"):
        if hasattr(latent_diffusion.first_stage_model, "vocoder"):
            print("Optimizing vocoder (removing weight norm)...")
            try:
                latent_diffusion.first_stage_model.vocoder.remove_weight_norm()
            except Exception as e:
                print(f"Failed to remove weight norm from vocoder: {e}")

    if compile:
        import time

        print(
            f"\n[AudioSR] Starting model compilation (warmup_duration={warmup_duration}s)..."
        )
        start_comp = time.perf_counter()
        try:
            # UNet compilation
            print("[AudioSR] Compiling UNet...")
            latent_diffusion.model.diffusion_model = torch.compile(
                latent_diffusion.model.diffusion_model
            )

            # Vocoder compilation
            if hasattr(latent_diffusion.first_stage_model, "vocoder"):
                print("[AudioSR] Compiling Vocoder...")
                latent_diffusion.first_stage_model.vocoder = torch.compile(
                    latent_diffusion.first_stage_model.vocoder
                )

            # Warm-up to trigger compilation NOW instead of during the first request
            # IMPORTANT: Duration MUST match the actual inference chunks to avoid re-compilation
            print(
                f"[AudioSR] Running compilation warm-up for {warmup_duration}s chunk..."
            )
            print(
                "[AudioSR] Note: This will take 2-4 minutes on the very first run (populating cache)."
            )

            warmup_start = time.perf_counter()
            with torch.no_grad():
                # Correctly size the dummy wave for the target duration
                dummy_wav = np.zeros((int(48000 * warmup_duration)), dtype=np.float32)
                batch, duration = make_batch_for_super_resolution(
                    None, waveform=dummy_wav
                )
                # Move batch to device
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)

                # Single step is enough to trigger JIT compilation of the UNet and Vocoder
                print("[AudioSR] Triggering first forward pass (JIT compiling)...")
                latent_diffusion.generate_batch(
                    batch,
                    unconditional_guidance_scale=1.0,
                    ddim_steps=1,
                    duration=duration,
                )

            warmup_end = time.perf_counter()
            print(
                f"✓ [AudioSR] Compilation warm-up successful (took {warmup_end - warmup_start:.2f}s)."
            )
            print(
                f"✓ [AudioSR] Total compilation time: {warmup_end - start_comp:.2f}s.\n"
            )
        except Exception as e:
            print(f"✗ [AudioSR] Failed to compile or warm-up model: {e}")
            import traceback

            traceback.print_exc()

    profiler.stop()
    print(profiler.output_text(show_all=True, unicode=True, color=True))

    return latent_diffusion


def super_resolution(
    latent_diffusion,
    input_file,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
    latent_t_per_second=12.8,
    config=None,
):
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()
    seed_everything(int(seed))
    waveform = None

    batch, duration = make_batch_for_super_resolution(input_file, waveform=waveform)

    with torch.no_grad():
        waveform = latent_diffusion.generate_batch(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            duration=duration,
        )

    profiler.stop()
    print(profiler.output_text(show_all=True, unicode=True, color=True))

    return waveform


def super_resolution_from_waveform(
    latent_diffusion,
    waveform_np,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
):
    """
    Process a single waveform directly without saving to file.

    Args:
        latent_diffusion: The model
        waveform_np: numpy array of shape (samples,) at 48kHz
        seed: Random seed
        ddim_steps: Number of DDIM steps
        guidance_scale: Guidance scale for generation

    Returns:
        Processed waveform as numpy array
    """
    seed_everything(int(seed))

    sampling_rate = 48000
    original_length = len(waveform_np)
    duration = original_length / sampling_rate

    # Pad to multiple of 5.12s
    if duration % 5.12 != 0:
        pad_duration = duration + (5.12 - duration % 5.12)
    else:
        pad_duration = duration

    target_frame = int(pad_duration * 100)

    # Normalize and pad
    waveform = normalize_wav(waveform_np)
    waveform = pad_wav(waveform, target_length=int(sampling_rate * pad_duration))

    # Extract features
    log_mel_spec, stft = wav_feature_extraction(
        torch.from_numpy(waveform), target_frame
    )

    batch = {
        "waveform": torch.FloatTensor(waveform),
        "stft": torch.FloatTensor(stft),
        "log_mel_spec": torch.FloatTensor(log_mel_spec),
        "sampling_rate": sampling_rate,
    }

    batch.update(lowpass_filtering_prepare_inference(batch))

    lowpass_mel, lowpass_stft = wav_feature_extraction(
        batch["waveform_lowpass"], target_frame
    )
    batch["lowpass_mel"] = lowpass_mel

    # Add batch dimension to all tensors
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = torch.FloatTensor(batch[k]).unsqueeze(0)

    # Process
    with torch.no_grad():
        result = latent_diffusion.generate_batch(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            duration=pad_duration,
        )

    # Convert to numpy and trim to original length
    if isinstance(result, torch.Tensor):
        result = result.cpu().numpy()

    result = np.squeeze(result)
    if len(result) > original_length:
        result = result[:original_length]

    return result


def _prepare_single_waveform_batch(waveform_np, target_duration, sampling_rate=48000):
    """
    Prepare batch tensors for a single waveform chunk.
    All chunks will be padded to target_duration (in seconds).

    Returns:
        batch: dict with tensors (no batch dimension yet)
        original_length: original sample count before padding
    """
    original_length = len(waveform_np)
    target_frame = int(target_duration * 100)
    target_samples = int(target_duration * sampling_rate)

    # Normalize and pad
    waveform = normalize_wav(waveform_np)
    waveform = pad_wav(waveform, target_length=target_samples)

    # Extract features
    log_mel_spec, stft = wav_feature_extraction(
        torch.from_numpy(waveform), target_frame
    )

    batch = {
        "waveform": torch.FloatTensor(waveform),
        "stft": torch.FloatTensor(stft),
        "log_mel_spec": torch.FloatTensor(log_mel_spec),
        "sampling_rate": sampling_rate,
    }

    batch.update(lowpass_filtering_prepare_inference(batch))

    lowpass_mel, lowpass_stft = wav_feature_extraction(
        batch["waveform_lowpass"], target_frame
    )
    batch["lowpass_mel"] = lowpass_mel

    return batch, original_length


def super_resolution_batch(
    latent_diffusion,
    waveforms_list,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
):
    """
    Process multiple waveform chunks in a SINGLE forward pass (true GPU parallelism).

    Args:
        latent_diffusion: The model
        waveforms_list: List of numpy arrays, each shape (samples,) at 48kHz
        seed: Random seed
        ddim_steps: Number of DDIM steps
        guidance_scale: Guidance scale for generation

    Returns:
        List of processed waveforms as numpy arrays (trimmed to original lengths)
    """
    if len(waveforms_list) == 0:
        return []

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    seed_everything(int(seed))
    sampling_rate = 48000

    # Find the maximum duration and round up to 5.12s multiple
    max_duration = max(len(w) / sampling_rate for w in waveforms_list)
    if max_duration % 5.12 != 0:
        target_duration = max_duration + (5.12 - max_duration % 5.12)
    else:
        target_duration = max_duration

    print(
        f"[Batch] Processing {len(waveforms_list)} chunks, target_duration={target_duration:.2f}s"
    )

    # Prepare batch for each waveform
    batch_list = []
    original_lengths = []
    for waveform_np in waveforms_list:
        batch, orig_len = _prepare_single_waveform_batch(
            waveform_np, target_duration, sampling_rate
        )
        batch_list.append(batch)
        original_lengths.append(orig_len)

    # Stack all batches along dim=0
    combined_batch = {}
    for key in batch_list[0].keys():
        if isinstance(batch_list[0][key], torch.Tensor):
            # Stack tensors: each has shape [time, ...] or [samples]
            # After stack, shape becomes [batch, time, ...] or [batch, samples]
            tensors = [b[key] for b in batch_list]
            combined_batch[key] = torch.stack(tensors, dim=0)
            print(f"  {key}: {combined_batch[key].shape}")
        else:
            combined_batch[key] = batch_list[0][key]

    # Process all chunks at once
    with torch.no_grad():
        results = latent_diffusion.generate_batch(
            combined_batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            duration=target_duration,
        )

    # Convert results to numpy
    if isinstance(results, torch.Tensor):
        results = results.cpu().numpy()

    # Split results back and trim to original lengths
    output_list = []
    for i, orig_len in enumerate(original_lengths):
        result = results[i]
        result = np.squeeze(result)
        if len(result) > orig_len:
            result = result[:orig_len]
        output_list.append(result)

    profiler.stop()
    print(profiler.output_text(show_all=True, unicode=True, color=True))

    return output_list


def super_resolution_long_audio(
    latent_diffusion,
    input_file,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
    chunk_duration_s=15,
    overlap_duration_s=2,
):
    """
    Processes a long audio file by chunking it, running super-resolution on each chunk,
    and reconstructing the full audio with cross-fading in overlap regions.
    """

    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    seed_everything(int(seed))

    if chunk_duration_s <= overlap_duration_s:
        raise ValueError("Chunk duration must be greater than overlap duration.")

    # 1. Load the entire audio file once
    # Use soundfile directly to avoid torchcodec issues in newer torchaudio versions
    waveform_np, sr = sf.read(input_file, dtype="float32")
    # soundfile returns (samples, channels), convert to (channels, samples) tensor
    if waveform_np.ndim == 1:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    else:
        waveform = torch.from_numpy(waveform_np.T)

    # Resample to 48000 Hz
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        waveform = resampler(waveform)
        sr = 48000

    # Ensure waveform is mono for processing
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.unsqueeze(0)  # Add a batch dimension

    # 2. Define chunk and overlap sizes in samples
    chunk_samples = int(chunk_duration_s * sr)
    overlap_samples = int(overlap_duration_s * sr)
    step_samples = chunk_samples - overlap_samples
    total_samples = waveform.shape[2]

    # Create a buffer for the final output
    final_waveform = torch.zeros_like(waveform)
    # Create a buffer to track overlap contributions for normalization
    overlap_contribution_map = torch.zeros_like(waveform)

    # 3. Create a linear fade-in/fade-out window for cross-fading
    fade_window = torch.hann_window(2 * overlap_samples, periodic=False)
    fade_in = fade_window[:overlap_samples]
    fade_out = fade_window[overlap_samples:]

    # 4. Iterate over chunks
    for start_sample in range(0, total_samples, step_samples):
        end_sample = start_sample + chunk_samples

        # Get the current chunk
        chunk_waveform = waveform[:, :, start_sample:end_sample]

        # *** NEW: Record the original peak amplitude of the chunk ***
        # Add a small epsilon to avoid division by zero for silent chunks
        original_peak = torch.max(torch.abs(chunk_waveform)) + 1e-8

        # Pad the last chunk if it's shorter than chunk_samples
        current_chunk_len = chunk_waveform.shape[2]
        if current_chunk_len < chunk_samples:
            padding_needed = chunk_samples - current_chunk_len
            chunk_waveform = F.pad(chunk_waveform, (0, padding_needed))

        print(
            f"Processing chunk from {start_sample / sr:.2f}s to {end_sample / sr:.2f}s"
        )

        # --- This part replaces the original `super_resolution` logic ---
        # Prepare batch from the waveform chunk directly
        batch, duration = make_batch_for_super_resolution(
            None, waveform=chunk_waveform.squeeze(0).numpy()
        )

        with torch.no_grad():
            # Use autocast if device is cuda or cpu. MPS autocast is limited.
            # Robustly get the device type
            try:
                device_obj = getattr(
                    latent_diffusion,
                    "device",
                    next(latent_diffusion.parameters()).device,
                )
                if isinstance(device_obj, str):
                    device_obj = torch.device(device_obj)
                autocast_device = (
                    "cuda"
                    if device_obj.type == "cuda"
                    else ("cpu" if device_obj.type == "cpu" else None)
                )
            except Exception:
                autocast_device = None

            if autocast_device:
                with torch.autocast(device_type=autocast_device):
                    # Run inference on the single chunk
                    processed_chunk = latent_diffusion.generate_batch(
                        batch,
                        unconditional_guidance_scale=guidance_scale,
                        ddim_steps=ddim_steps,
                        duration=duration,
                    )
            else:
                # Run inference on the single chunk
                processed_chunk = latent_diffusion.generate_batch(
                    batch,
                    unconditional_guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    duration=duration,
                )

        # Ensure the processed chunk is a tensor
        if isinstance(processed_chunk, np.ndarray):
            processed_chunk = torch.from_numpy(processed_chunk)

        # Trim padding from the last chunk if necessary
        processed_chunk = processed_chunk[:, :, :current_chunk_len]

        # *** NEW: Rescale the output chunk to match the original peak volume ***
        processed_peak = torch.max(torch.abs(processed_chunk)) + 1e-8
        # Apply the scaling factor
        processed_chunk = (processed_chunk / processed_peak) * original_peak

        # 5. Apply cross-fading window to the overlap regions
        # The very first chunk has no left overlap to fade in
        if start_sample > 0:
            processed_chunk[:, :, :overlap_samples] *= fade_in

        # The very last chunk has no right overlap to fade out
        if end_sample < total_samples:
            processed_chunk[:, :, -overlap_samples:] *= fade_out

        # 6. Add the processed chunk to the final waveform (Overlap-Add)
        final_waveform[:, :, start_sample:end_sample] += processed_chunk.to(
            final_waveform.device
        )

        # Update the contribution map for normalization
        window_contribution = torch.ones(current_chunk_len)
        if start_sample > 0:
            window_contribution[:overlap_samples] = fade_in
        if end_sample < total_samples:
            window_contribution[-overlap_samples:] = fade_out
        overlap_contribution_map[:, :, start_sample:end_sample] += (
            window_contribution.to(overlap_contribution_map.device)
        )

        # Clean up memory - removed empty_cache/gc as they are slow
        del batch, processed_chunk

    # 7. Normalize the overlapping regions
    # Avoid division by zero in non-overlapping parts
    overlap_contribution_map[overlap_contribution_map == 0] = 1.0
    final_waveform /= overlap_contribution_map

    # Clamp the final output to avoid clipping
    final_waveform = torch.clamp(final_waveform, -1.0, 1.0)

    profiler.stop()
    print(profiler.output_text(show_all=True, unicode=True, color=True))

    return final_waveform.squeeze(0)  # Remove batch dimension before saving
