import pytest
import torch

from audiosr import pipeline


def test_lowpass_by_downsampling_uses_explicit_sample_rate(
    monkeypatch,
) -> None:
    resample_calls: list[tuple[int, int]] = []

    def fake_resample(
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int,
    ) -> torch.Tensor:
        resample_calls.append((orig_freq, new_freq))
        return waveform.clone()

    monkeypatch.setattr(pipeline.torchaudio.functional, "resample", fake_resample)

    waveform = torch.zeros(1, 48000)
    filtered = pipeline.lowpass_by_downsampling(
        waveform,
        sampling_rate=48000,
        lowpass_sampling_rate_hz=9000,
    )

    assert filtered.shape == waveform.shape
    assert resample_calls == [(48000, 9000), (9000, 48000)]


@pytest.mark.parametrize("length", [356, 1024])
def test_assemble_original_signal_low_band_handles_short_waveforms(
    length: int,
) -> None:
    waveform = torch.linspace(-0.25, 0.25, length)

    assembled = pipeline.assemble_original_signal_low_band(
        generated_audio=waveform,
        original_audio=waveform,
        cutoff_hz=3000,
    )

    assert assembled.shape == waveform.shape
    assert torch.isfinite(assembled).all()
    torch.testing.assert_close(assembled, waveform, atol=1e-6, rtol=1e-5)


class _EchoLatentDiffusion:
    def __init__(self) -> None:
        self.chunk_lengths: list[int] = []

    def generate_batch(self, batch, **_kwargs):
        self.chunk_lengths.append(batch["waveform"].shape[-1])
        return batch["waveform"].unsqueeze(0).numpy()


# 0.125 s chunks with 0.03125 s overlap are 6000 and 1500 samples at 48 kHz,
# so the stride is 4500 samples.
@pytest.mark.parametrize(
    ("total_samples", "expected_calls"),
    [
        (4500 + 757, 1),
        (6000, 1),
        (6001, 2),
        (15001, 4),
    ],
)
def test_super_resolution_long_audio_only_runs_chunks_past_previous_end(
    tmp_path,
    monkeypatch,
    total_samples: int,
    expected_calls: int,
) -> None:
    import numpy as np
    import soundfile as sf

    def fake_make_batch(_input_file, waveform=None, fbank=None):
        return {"waveform": torch.from_numpy(waveform)}, waveform.shape[-1] / 48000

    monkeypatch.setattr(pipeline, "make_batch_for_super_resolution", fake_make_batch)

    time = np.arange(total_samples, dtype=np.float32) / 48000
    waveform = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    input_path = tmp_path / "input.wav"
    sf.write(input_path, waveform, 48000, subtype="FLOAT")
    model = _EchoLatentDiffusion()

    output = pipeline.super_resolution_long_audio(
        model,
        str(input_path),
        chunk_duration_s=0.125,
        overlap_duration_s=0.03125,
    )

    assert len(model.chunk_lengths) == expected_calls
    assert set(model.chunk_lengths) == {6000}
    assert output.shape == (1, total_samples)
    torch.testing.assert_close(
        output.squeeze(0), torch.from_numpy(waveform), atol=1e-5, rtol=1e-5
    )
