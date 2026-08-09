"""CPU checks for scalar-equivalent and independent hard-side queries."""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from split_voxelize import split_cell_query


def fixture():
    points = torch.zeros(1, 3)
    query = torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]], dtype=torch.float32)
    nn_idx = torch.zeros(2, dtype=torch.long)
    density = torch.tensor([[0.2]])
    delta = torch.zeros(1, 1)
    quat = torch.tensor([[1.0, 0, 0, 0]])
    sites = torch.tensor([[[.4, 0], [0, .4], [-.4, 0], [0, -.4]]])
    heights = torch.zeros(1, 4)
    radius = torch.ones(1)
    return query, points, nn_idx, density, delta, quat, sites, heights, radius


def query(mode, raw_plus=None, raw_minus=None, delta=None):
    q, p, nn, density, zero, quat, sites, heights, radius = fixture()
    return split_cell_query(
        q, p, nn, density, zero if delta is None else delta,
        quat, sites, heights, radius, density_mode=mode,
        raw_plus=raw_plus, raw_minus=raw_minus,
        delta_max_frac=.5,
    )[0]


def main():
    _, _, _, density, _, _, _, _, _ = fixture()
    scalar = torch.nn.functional.softplus(density.squeeze(-1), beta=10)[0]
    absolute0 = query("absolute")
    relative0 = query("relative")
    independent0 = query("independent", density.clone(), density.clone())
    assert torch.allclose(absolute0, torch.full_like(absolute0, scalar), atol=1e-7)
    assert torch.allclose(relative0, absolute0, atol=1e-7)
    assert torch.allclose(independent0, absolute0, atol=1e-7)

    rp = torch.tensor([[0.7]])
    rm = torch.tensor([[-0.4]])
    values = query("independent", rp, rm)
    expected_plus = torch.nn.functional.softplus(rp, beta=10).item()
    expected_minus = torch.nn.functional.softplus(rm, beta=10).item()
    assert abs(values[0].item() - expected_plus) < 1e-7
    assert abs(values[1].item() - expected_minus) < 1e-7
    assert values.min() >= 0

    raw_delta = torch.tensor([[0.8]])
    rel = query("relative", delta=raw_delta)
    d = .5 * scalar * torch.tanh(raw_delta[0, 0])
    assert torch.allclose(rel, torch.tensor([scalar + d, scalar - d]), atol=1e-7)

    try:
        query("independent")
    except ValueError:
        pass
    else:
        raise AssertionError("independent query accepted missing raw sides")
    print("SUMMARY: ALL SPLIT-VOXELIZE DENSITY-MODE TESTS PASSED")


if __name__ == "__main__":
    main()
