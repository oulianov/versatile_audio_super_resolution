import os
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pytest
import torch

from audiosr import utils
from audiosr.utilities.audio.audio_processing import window_sumsquare
from audiosr.utilities.audio.stft import STFT, TacotronSTFT


def test_inference_import_and_preprocessing_without_librosa(tmp_path: Path) -> None:
    script = """
import importlib.abc
import sys

class BlockLibrosa(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'librosa', 'torchlibrosa', 'numba'}:
            raise AssertionError(f'Inference attempted to import {fullname}')

sys.meta_path.insert(0, BlockLibrosa())
import audiosr
import soundfile as sf
import torch
from audiosr.latent_encoder.autoencoder import AutoencoderKL
from audiosr.latent_diffusion.modules.encoders.modules import VAEFeatureExtract
from audiosr.utils import mel_spectrogram_train, read_audio_file

waveform = torch.linspace(-0.4, 0.4, 48000)
sf.write(sys.argv[1], waveform.numpy(), 48000)
audio = read_audio_file(sys.argv[1])
mel, spectrum = mel_spectrogram_train(waveform.unsqueeze(0))
assert mel.shape[0] == 256
assert torch.isfinite(mel).all()
assert spectrum.shape[0] == 1025
assert not any(name.split('.')[0] in {'librosa', 'torchlibrosa', 'numba'} for name in sys.modules)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "input.wav")],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("signal", ["noise", "silence", "tone", "impulse"])
def test_inference_log_mel_preserves_librosa_preprocessing(signal: str) -> None:
    rng = torch.Generator().manual_seed(42)
    waveform = torch.randn(1, 48000, generator=rng) * 0.1
    if signal == "silence":
        waveform.zero_()
    elif signal == "tone":
        waveform = (
            0.2 * torch.sin(torch.arange(48000) * (2 * torch.pi * 440 / 48000))
        ).unsqueeze(0)
    elif signal == "impulse":
        waveform.zero_()
        waveform[0, 24000] = 0.5
    actual, spectrum = utils.mel_spectrogram_train(waveform)
    reference_basis = torch.from_numpy(
        librosa.filters.mel(sr=48000, n_fft=2048, n_mels=256, fmin=20, fmax=24000)
    )
    expected = torch.log(torch.clamp(reference_basis @ spectrum, min=1e-5))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("norm", [None, 0, 1, 2, np.inf, -np.inf])
@pytest.mark.parametrize("win_length", [47, 64])
def test_window_envelope_matches_librosa(norm: float | None, win_length: int) -> None:
    actual = window_sumsquare("hann", 9, 16, win_length, 64, norm=norm)
    expected = librosa.filters.window_sumsquare(
        window="hann",
        n_frames=9,
        hop_length=16,
        win_length=win_length,
        n_fft=64,
        norm=norm,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_stft_padding_and_inverse_preserve_waveform() -> None:
    waveform = torch.randn(1, 256, generator=torch.Generator().manual_seed(42)) * 0.1
    stft = STFT(filter_length=64, hop_length=16, win_length=47)
    magnitude, phase = stft.transform(waveform)
    expected = librosa.stft(
        waveform.numpy()[0], n_fft=64, hop_length=16, win_length=47, pad_mode="reflect"
    )
    torch.testing.assert_close(
        magnitude[0], torch.from_numpy(np.abs(expected)), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        stft.inverse(magnitude, phase).squeeze(1), waveform, rtol=1e-5, atol=1e-6
    )


def test_tacotron_mel_uses_slaney_scale_and_normalization() -> None:
    transform = TacotronSTFT(256, 64, 255, 16, 16000, 20, None)
    waveform = torch.randn(1, 1024, generator=torch.Generator().manual_seed(42)) * 0.1
    actual, magnitude, _, _ = transform.mel_spectrogram(waveform)
    reference_basis = torch.from_numpy(
        librosa.filters.mel(sr=16000, n_fft=256, n_mels=16, fmin=20)
    )
    expected = torch.log(torch.clamp(reference_basis @ magnitude, min=1e-5))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
