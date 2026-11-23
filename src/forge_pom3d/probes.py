"""Projected spatial harmonic action probes and network action-Jacobian rank diagnostics."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .network import ForgePOMNet
from .physics_env import BlackBoxProvider3D

HADAMARD_4X4: tuple[tuple[float, float, float, float], ...] = (
    (1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0, 1.0),
)


@torch.no_grad()
def measure_action_jacobian_ranks(
    model: ForgePOMNet,
    directions: list[torch.Tensor],
    obs_samples: list[torch.Tensor],
    mask_60: torch.Tensor,
    eps: float = 1e-2,
) -> dict[str, int]:
    """Measure effective numerical rank of first-order action responses for full vs trunk-only directions."""
    if not obs_samples or not directions:
        return {"action_jac_rank_full": 0, "action_jac_rank_body_only": 0}
    was_training = model.training
    model.eval()
    theta0 = model.get_flat_params()
    _, body_mask = model.head_body_index_masks(device=theta0.device)
    body_mask_f = body_mask.to(dtype=theta0.dtype)
    xb = torch.stack(obs_samples[:4], dim=0)
    m_b = mask_60.unsqueeze(0).double()

    full_cols: list[torch.Tensor] = []
    body_cols: list[torch.Tensor] = []
    for u_m in directions:
        model.set_flat_params(theta0 + eps * u_m)
        gp = model(xb).double()
        model.set_flat_params(theta0 - eps * u_m)
        gm = model(xb).double()
        full_cols.append(((gp - gm) * m_b / (2.0 * eps)).reshape(-1))

        u_body = u_m * body_mask_f
        model.set_flat_params(theta0 + eps * u_body)
        bp = model(xb).double()
        model.set_flat_params(theta0 - eps * u_body)
        bm = model(xb).double()
        body_cols.append(((bp - bm) * m_b / (2.0 * eps)).reshape(-1))

    model.set_flat_params(theta0)
    if was_training:
        model.train()

    def _num_rank(cols: list[torch.Tensor]) -> int:
        mat = torch.stack(cols, dim=1)
        gram = mat.T @ mat
        if dist.is_initialized():
            dist.all_reduce(gram, op=dist.ReduceOp.AVG)
        evals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        top = float(evals.max().item())
        if top <= 1e-20:
            return 0
        return int((evals > top * 1e-8).sum().item())

    return {
        "action_jac_rank_full": _num_rank(full_cols),
        "action_jac_rank_body_only": _num_rank(body_cols),
    }


@torch.no_grad()
def build_projected_action_probe_direction(
    env: BlackBoxProvider3D,
    mask_60: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct a unit-wet-RMS projected standardized action direction spanning channels & frequencies."""
    im, jm = env.cfg.im, env.cfg.jm
    in_ch = env.cfg.out_channels
    wet_cnt = float(mask_60.double().sum().clamp_min(1.0).item())
    x_coord = torch.linspace(0.0, math.pi, im, dtype=torch.float32).unsqueeze(1).expand(im, jm)
    y_coord = torch.linspace(0.0, math.pi, jm, dtype=torch.float32).unsqueeze(0).expand(im, jm)

    for attempt in range(8):
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + attempt * 7919)
        raw_cpu = torch.zeros((in_ch, im, jm), dtype=torch.float32)

        harm = torch.zeros((im, jm), dtype=torch.float32)
        coeffs_basin = torch.randn((4, 4), generator=gen, dtype=torch.float32)
        for px in range(1, 5):
            for py in range(1, 5):
                harm = harm + coeffs_basin[px - 1, py - 1] * torch.sin(
                    float(px) * x_coord
                ) * torch.sin(float(py) * y_coord)
        chan_weights = torch.randn((in_ch, 1, 1), generator=gen, dtype=torch.float32)
        chan_coherent = 0.5 + 0.5 * chan_weights
        raw_cpu = raw_cpu + 0.6 * chan_coherent * harm.unsqueeze(0)

        for _mode_idx in range(6):
            kx = int(torch.randint(1, 12, (1,), generator=gen).item())
            ky = int(torch.randint(1, 12, (1,), generator=gen).item())
            ph_x = float(torch.rand((1,), generator=gen).item()) * math.pi
            ph_y = float(torch.rand((1,), generator=gen).item()) * math.pi
            wave = torch.sin(float(kx) * x_coord + ph_x) * torch.sin(float(ky) * y_coord + ph_y)
            c_vec = torch.randn((in_ch, 1, 1), generator=gen, dtype=torch.float32)
            raw_cpu = raw_cpu + 0.4 * c_vec * wave.unsqueeze(0)

        raw_cpu = raw_cpu + 0.35 * torch.randn((in_ch, im, jm), generator=gen, dtype=torch.float32)
        proj = env.project_standardized_action(raw_cpu.to(device=device)) * mask_60
        rms = math.sqrt(float(proj.double().square().sum().item()) / wet_cnt)
        if rms > 1e-4:
            return (proj / float(rms)).contiguous()

    raise RuntimeError(f"Failed to construct non-zero projected probe direction for seed={seed}")


__all__ = [
    "HADAMARD_4X4",
    "build_projected_action_probe_direction",
    "measure_action_jacobian_ranks",
]
