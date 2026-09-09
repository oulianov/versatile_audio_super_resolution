import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Load the vocoder independently of audiosr's eager full-pipeline imports.
spec = importlib.util.spec_from_file_location(
    "audiosr_vocoder", Path(__file__).parents[1] / "audiosr/hifigan/models.py"
)
assert spec is not None and spec.loader is not None
vocoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vocoder)


@pytest.mark.parametrize("mel_bins", [64, 256])
def test_plain_vocoder_loads_existing_weights_and_preserves_audio(
    monkeypatch: pytest.MonkeyPatch, mel_bins: int
) -> None:
    config = SimpleNamespace(
        num_mels=mel_bins,
        upsample_initial_channel=16,
        upsample_rates=[2, 2],
        upsample_kernel_sizes=[4, 4],
        resblock_kernel_sizes=[3, 7],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
    )
    original = vocoder.Generator(config).eval()
    original.remove_weight_norm()

    def reject_normalization(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Inference construction must not apply weight normalization"
        )

    monkeypatch.setattr(vocoder, "weight_norm", reject_normalization)
    inference = vocoder.Generator(config, use_weight_norm=False).eval()
    inference.load_state_dict(original.state_dict(), strict=True)
    generator = torch.Generator(device="cpu").manual_seed(42)
    with torch.inference_mode():
        for frames in (1, 9, 32):
            mel = torch.randn(2, mel_bins, frames, generator=generator)
            torch.testing.assert_close(inference(mel), original(mel), rtol=0, atol=0)
