import math
from types import SimpleNamespace

import torch

from audiosr import pipeline
from audiosr.latent_diffusion.models import ddim
from audiosr.latent_diffusion.modules.diffusionmodules.model import AttnBlock


def test_window_starts_cover_the_track_with_equal_windows() -> None:
    assert pipeline.window_starts(10, 4, 3) == [0, 3, 6]
    assert pipeline.window_starts(11, 4, 3) == [0, 3, 6, 9]
    assert pipeline.window_starts(3, 4, 3) == [0]


def test_merging_slices_of_one_signal_returns_that_signal() -> None:
    # Windows cut from one signal agree on every overlap, so merging them must
    # give the signal back unchanged.
    canvas = torch.randn(16, 40, 8)
    starts = pipeline.window_starts(40, 16, 12)
    windows = torch.stack([canvas[:, start : start + 16] for start in starts])
    weights = pipeline.window_merge_weights(starts, 16)

    merged = pipeline.overlap_add(windows, starts, weights, 40, time_dim=1)

    torch.testing.assert_close(merged, canvas)


def test_disagreeing_windows_crossfade_across_their_overlap() -> None:
    starts = [0, 6]
    windows = torch.stack([torch.zeros(10), torch.ones(10)])
    weights = pipeline.window_merge_weights(starts, 10)

    merged = pipeline.overlap_add(windows, starts, weights, 16, time_dim=0)

    assert torch.all(merged[:6] == 0)
    assert torch.all(merged[10:] == 1)
    overlap = merged[6:10]
    assert torch.all((overlap > 0) & (overlap < 1))
    assert torch.all(overlap[1:] > overlap[:-1])


def test_short_crossfade_switches_windows_inside_a_longer_overlap() -> None:
    # Windows overlap by 8 samples but crossfade over the middle 2 only.
    starts = [0, 8]
    windows = torch.stack([torch.zeros(16), torch.ones(16)])
    weights = pipeline.window_merge_weights(starts, 16, crossfade=2)

    merged = pipeline.overlap_add(windows, starts, weights, 24, time_dim=0)

    assert torch.all(merged[:11] == 0)
    assert torch.all(merged[13:] == 1)
    assert 0 < merged[11] < merged[12] < 1


def test_angles_map_back_to_their_diffusion_steps() -> None:
    steps = torch.arange(1000, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / 1000 + 0.008) / 1.008 * math.pi / 2) ** 2
    table = torch.atan2((1 - alphas_cumprod).sqrt(), alphas_cumprod.sqrt()).float()

    on_grid = ddim.timesteps_for_angles(table[[0, 250, 999]], table)
    between = ddim.timesteps_for_angles((table[[10]] + table[[11]]) / 2, table)

    torch.testing.assert_close(on_grid, torch.tensor([0.0, 250.0, 999.0]))
    assert 10 < float(between[0]) < 11


def test_midpoint_solver_integrates_an_exponential_decay() -> None:
    state = ddim.solve_midpoint(lambda value, time: -value, torch.ones(3), num_steps=20)

    torch.testing.assert_close(state, torch.full((3,), math.exp(-1)), rtol=1e-3, atol=0)


def test_probability_flow_follows_the_exact_trajectory_to_the_data() -> None:
    # A model that predicts v exactly for one data point: the ODE must land
    # where z = cos(angle) * x0 + sin(angle) * eps puts it at the last step.
    steps = torch.arange(1000, dtype=torch.float64)
    alphas_cumprod = torch.cos((steps / 1000 + 0.008) / 1.008 * math.pi / 2) ** 2
    model = SimpleNamespace(
        parameterization="v",
        sqrt_alphas_cumprod=alphas_cumprod.sqrt().float(),
        sqrt_one_minus_alphas_cumprod=(1 - alphas_cumprod).sqrt().float(),
    )
    table = torch.atan2(model.sqrt_one_minus_alphas_cumprod, model.sqrt_alphas_cumprod)
    x0 = torch.tensor([0.5, -1.0, 2.0])

    def angle_at(timestep: float) -> torch.Tensor:
        lower = min(int(timestep), 998)
        return table[lower] + (timestep - lower) * (table[lower + 1] - table[lower])

    def exact_v(z: torch.Tensor, timestep: float) -> torch.Tensor:
        angle = angle_at(timestep)
        eps = (z - torch.cos(angle) * x0) / torch.sin(angle)
        return torch.cos(angle) * eps - torch.sin(angle) * x0

    noise = torch.tensor([1.2, -0.3, 0.7])
    eps = (noise - torch.cos(table[-1]) * x0) / torch.sin(table[-1])
    expected = torch.cos(table[0]) * x0 + torch.sin(table[0]) * eps

    result = ddim.solve_probability_flow(model, exact_v, noise, num_steps=50)

    torch.testing.assert_close(result, expected, atol=1e-3, rtol=0)


def test_track_chunks_cover_the_track_without_trailing_silence() -> None:
    hop = pipeline.MEL_HOP_SAMPLES
    window, overlap = 2048 * hop, 1920
    for total in (window, window + 1, 3 * window - 12345, 60 * 48000):
        starts = pipeline.track_chunk_starts(total, window, overlap)

        assert starts[0] == 0
        assert all(start % hop == 0 for start in starts)
        assert all(b - a <= window - overlap for a, b in zip(starts, starts[1:]))
        assert 0 <= starts[-1] + window - total < hop


def test_vae_attention_matches_softmax_attention() -> None:
    torch.manual_seed(0)
    block = AttnBlock(32).eval()
    x = torch.randn(2, 32, 3, 5)

    with torch.no_grad():
        h = block.norm(x)
        q, k, v = (layer(h).flatten(2) for layer in (block.q, block.k, block.v))
        weights = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(32), dim=-1)
        expected = x + block.proj_out((v @ weights.transpose(1, 2)).view_as(x))

        torch.testing.assert_close(block(x), expected)
