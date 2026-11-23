"""Shared action exploration spectra, coherent DCT probe schedules, and spatial cost math."""

from __future__ import annotations

import numpy as np


def spectrum(n, m, floor=0.05):
    ky = np.fft.fftfreq(n) * n
    kx = np.fft.rfftfreq(m) * m
    scale = np.where((np.abs(ky[:, None]) < 6) & (kx[None, :] < 6), 1.0, float(floor))
    weights = np.full(m // 2 + 1, 2.0)
    weights[0] = 1.0
    if m % 2 == 0:
        weights[-1] = 1.0
    return scale / np.sqrt((scale**2 * weights).sum() / (n * m))


def smooth_noise(rng, shape, floor=0.05):
    white = rng.normal(size=shape)
    return np.fft.irfft2(
        np.fft.rfft2(white, norm="ortho") * spectrum(*shape[-2:], floor=floor),
        s=shape[-2:],
        norm="ortho",
    ).astype(np.float32)


def orthogonal_temporal_basis(horizon, kind="dct2"):
    """Return horizon x horizon exact orthogonal temporal basis scaled to unit RMS per column."""
    t = np.arange(horizon, dtype=np.float64)[:, None]
    m = np.arange(horizon, dtype=np.float64)[None, :]
    if kind == "dct4":
        basis = np.sqrt(2.0) * np.cos(np.pi * (2.0 * t + 1.0) * (2.0 * m + 1.0) / (4.0 * horizon))
        return basis.astype(np.float32)
    basis = np.cos(np.pi * (2.0 * t + 1.0) * m / (2.0 * horizon))
    basis[:, 0] *= np.sqrt(1.0 / horizon)
    if horizon > 1:
        basis[:, 1:] *= np.sqrt(2.0 / horizon)
    return (basis * np.sqrt(float(horizon))).astype(np.float32)


def coherent_probe_schedule(rng_list, seed, horizon, action_shape, sigma):
    """Synthesize interval-coherent probes by combining spatial/channel directions with
    exact 20x20 orthogonal temporal bases (DCT-II for fit, DCT-IV for held-out), using
    frequency-stratified column partitions. No field mean or action is projected.
    """
    members = len(rng_list)
    if horizon < 1 or members < 1:
        raise ValueError("positive horizon and nonempty public batch required")
    if not sigma:
        return np.zeros((horizon, members, *action_shape), dtype=np.float32)
    slot = int(seed) % 10
    is_held = slot in (7, 8, 9)
    basis = orthogonal_temporal_basis(horizon, kind="dct4" if is_held else "dct2")
    variant = 0 if slot in (0, 7) else 1
    schedule = np.zeros((horizon, members, *action_shape), dtype=np.float32)
    modes_per_member = max(1, int(np.ceil(horizon / members)))
    for q in range(members):
        if not is_held:
            cols = [
                (members * r + ((q + 2 * variant) % members)) % horizon
                for r in range(modes_per_member)
            ]
        else:
            cols = [
                (members * r + ((members - 1 - q + variant) % members)) % horizon
                for r in range(modes_per_member)
            ]
        spatial_dirs = np.stack(
            [smooth_noise(rng_list[q], action_shape, floor=0.0) for _ in cols], axis=0
        )
        weights = basis[:, cols] / np.sqrt(float(len(cols)))
        member_sched = float(sigma) * np.einsum("tm,mchw->tchw", weights, spatial_dirs)
        schedule[:, q] = member_sched.astype(np.float32)
    return schedule


def spatial_cost(field, truth, scale):
    """Full-resolution cost whose spatial mean equals the capped native score."""
    difference = (np.asarray(field) - np.asarray(truth)) / np.asarray(scale).reshape(1, -1, 1, 1)
    cost = np.square(difference).mean(1)
    if not np.isfinite(cost).all():
        raise FloatingPointError("nonfinite native PPO cost")
    mean = cost.mean((1, 2))
    cost *= np.minimum(1.0, 200.0 / np.maximum(mean, 1e-30))[:, None, None]
    return cost.astype(np.float32)


__all__ = [
    "coherent_probe_schedule",
    "orthogonal_temporal_basis",
    "smooth_noise",
    "spatial_cost",
    "spectrum",
]
