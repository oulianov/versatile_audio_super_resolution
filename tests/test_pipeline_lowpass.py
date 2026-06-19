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
