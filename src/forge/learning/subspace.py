"""Band-stratified readout subspaces, body direction streams, and coordinate GN moments."""

from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call

from ..core.io import load_array
from .common import array, set_model
from .distributed import engine


@torch.no_grad()
def head_map(model, theta, obs):
    """The component network's linear decoder; every linear head column including bias is included."""
    prior = array(model).copy()
    set_model(model, theta)
    features = []
    handle = model.network.head.register_forward_pre_hook(
        lambda _, args: features.append(args[0].detach())
    )
    dev = next(model.parameters()).device
    result = []
    try:
        for o in obs:
            model(torch.as_tensor(o[None], device=dev))
            value = features.pop()[0].flatten(1).T
            value = torch.cat((value, torch.ones_like(value[:, :1])), dim=1)
            channels = model.action_channels
            width = value.shape[1] - 1
            decoder = value.new_zeros((channels, len(value), channels * (width + 1)))
            for channel in range(channels):
                decoder[channel, :, channel * width : (channel + 1) * width] = value[:, :width]
                decoder[channel, :, channels * width + channel] = 1
            value = decoder.flatten(0, 1) * model.amplitude
            result.append(value.cpu().double().numpy())
    finally:
        handle.remove()
        set_model(model, prior)
    return np.concatenate(result) / np.sqrt(len(obs) * obs.shape[-1] * obs.shape[-2])


def body_candidates(center, slices, round_index, slot):
    """Layer/mode/channel-local directions across lift, spectral, pointwise and project."""
    rng = np.random.default_rng(np.random.SeedSequence([4900, round_index, slot]))
    mapping = {name: (sl, shape) for name, sl, shape in slices}
    for retry in range(12):
        layer = (round_index + slot + slot // 4 + retry) % 4
        direction = np.zeros_like(center)
        if retry == 0 and slot in (6, 7) and "network.lift.weight" in mapping:
            prefix = "network.lift" if slot == 6 else "network.project"
            tag = "lift" if slot == 6 else "project"
            sl, shape = mapping[f"{prefix}.weight"]
            weight = center[sl].reshape(shape)
            a, b = rng.choice(shape[0], 2, replace=False)
            value = np.zeros(shape, np.float32)
            value[a] = weight[b]
            value[b] = -weight[a]
            direction[sl] = value.reshape(-1)
            bs, bshape = mapping[f"{prefix}.bias"]
            value = np.zeros(bshape, np.float32)
            value[a] = center[bs][b]
            value[b] = -center[bs][a]
            direction[bs] = value
            label = f"body_{tag}_L{layer}_channels_{a}_{b}"
        elif slot % 4 == 3:
            key = f"network.pointwise.{layer}.weight"
            sl, shape = mapping[key]
            weight = center[sl].reshape(shape)
            a, b = rng.choice(shape[0], 2, replace=False)
            value = np.zeros(shape, np.float32)
            value[a] = weight[b]
            value[b] = -weight[a]
            direction[sl] = value.reshape(-1)
            bias_key = f"network.pointwise.{layer}.bias"
            bs, bshape = mapping[bias_key]
            value = np.zeros(bshape, np.float32)
            value[a] = center[bs][b]
            value[b] = -center[bs][a]
            direction[bs] = value
            label = f"body_pointwise_L{layer}_channels_{a}_{b}"
        else:
            region = ("positive", "negative")[int(rng.integers(2))]
            key = f"network.spectral.{layer}.{region}"
            sl, shape = mapping[key]
            weight = center[sl].reshape(shape)
            iy = int(rng.integers(shape[2]))
            ix = int(rng.integers(shape[3]))
            rows = rng.choice(shape[0], min(4, shape[0]), replace=False)
            value = np.zeros(shape, np.float32)
            mode = (round_index + slot + retry) % 3
            if mode == 0:
                value[rows, :, iy, ix, 0] = weight[rows, :, iy, ix, 0]
            elif mode == 1:
                value[rows, :, iy, ix, 1] = weight[rows, :, iy, ix, 1]
            else:
                value[rows, :, iy, ix, 0] = -weight[rows, :, iy, ix, 1]
                value[rows, :, iy, ix, 1] = weight[rows, :, iy, ix, 0]
            direction[sl] = value.reshape(-1)
            label = (
                f"body_spectral_L{layer}_{region}_mode_{iy}_{ix}_"
                + ("real_gain", "imag_gain", "phase")[mode]
            )
        yield direction, label


def dense_body_candidates(center, slices, round_index, slot, seed=0):
    """Dimension- and scale-balanced dense additive body directions."""
    body_slices = [(name, sl, shape) for name, sl, shape in slices if ".head." not in name]
    num_tensors = len(body_slices)
    if num_tensors == 0:
        return
    inv_sqrt_b = 1.0 / np.sqrt(float(num_tensors))
    for retry in range(12):
        rng = np.random.default_rng(
            np.random.SeedSequence([4910, int(seed), int(round_index), int(slot), int(retry)])
        )
        direction = np.zeros(len(center), dtype=np.float32)
        for _, sl, _ in body_slices:
            b_ell = np.asarray(center[sl], dtype=np.float64).reshape(-1)
            s_ell = max(float(np.linalg.norm(b_ell)), 1e-2)
            g_ell = rng.normal(size=len(b_ell))
            g_norm = float(np.linalg.norm(g_ell))
            while g_norm <= 1e-30:
                g_ell = rng.normal(size=len(b_ell))
                g_norm = float(np.linalg.norm(g_ell))
            d_ell = (s_ell * inv_sqrt_b) * (g_ell / g_norm)
            direction[sl] = d_ell.astype(np.float32)
        label = f"body_dense_slot_{slot}" if retry == 0 else f"body_dense_slot_{slot}_r{retry}"
        yield direction, label


def anchor_forward(model, observations, slices):
    device = next(model.parameters()).device
    obs = torch.as_tensor(observations, device=device)
    scale = np.sqrt(len(obs) * obs.shape[-2] * obs.shape[-1])

    def forward(theta):
        parameters = {name: theta[sl].reshape(shape) for name, sl, shape in slices}
        return (
            torch.cat(
                [functional_call(model, parameters, (batch,)).flatten() for batch in obs.split(8)]
            )
            / scale
        )

    return forward


def anchor_output(model, theta, observations, slices):
    forward = anchor_forward(model, observations, slices)
    with torch.no_grad():
        return (
            forward(torch.as_tensor(theta, device=next(model.parameters()).device))
            .cpu()
            .double()
            .numpy()
        )


def _band_head_basis(model, features, seed_or_round):
    """Build readout basis respecting BandSplit2d blocks and action channels."""
    has_band_split = hasattr(model, "network") and hasattr(model.network, "band_split")
    channels = int(getattr(model, "action_channels", 1))
    band_groups = model.network.band_split.band_channels() if has_band_split else None
    if (
        band_groups is not None
        and (len(band_groups) > 1 or channels > 1)
        and features.shape[1] % channels == 0
    ):
        width = (features.shape[1] // channels) - 1
        diag_max = max(float(np.sum(features * features, axis=0).max()), 1e-30)
        modes_by_pair = {}
        vecs_list = []
        vals_list = []
        pairs_bc = [(b_idx, ch) for ch in range(channels) for b_idx in range(len(band_groups))]
        for b_idx, ch in pairs_bc:
            cols = [ch * width + c for c in band_groups[b_idx] if c < width]
            if b_idx == 0:
                cols.append(channels * width + ch)
            if not cols:
                modes_by_pair[(b_idx, ch)] = []
                continue
            sub = features[:, cols]
            ev, evec = np.linalg.eigh(sub.T @ sub)
            thresh = max(diag_max * 1e-12, float(ev.max()) * 1e-10, 1e-20)
            local_ids = []
            for m in np.argsort(ev)[::-1]:
                if ev[m] > thresh:
                    full_v = np.zeros(features.shape[1], dtype=np.float64)
                    full_v[cols] = evec[:, m]
                    idx = len(vals_list)
                    vecs_list.append(full_v)
                    vals_list.append(float(ev[m]))
                    local_ids.append(idx)
            modes_by_pair[(b_idx, ch)] = local_ids
        if len(vals_list) >= 16:
            eigenvalues = np.asarray(vals_list, dtype=np.float64)
            eigenvectors = np.stack(vecs_list, axis=1)
            order = [int(i) for i in np.argsort(eigenvalues)[::-1]]
            max_rank = max((len(v) for v in modes_by_pair.values()), default=0)
            stratified = [
                modes_by_pair[pair_key][rank_idx]
                for rank_idx in range(max_rank)
                for pair_key in pairs_bc
                if rank_idx < len(modes_by_pair[pair_key])
            ]
            top_pc = order[0]
            pool = [pc for pc in stratified if pc != top_pc]
            offset = (6 * int(seed_or_round)) % len(pool)
            selected = [top_pc] + pool[offset:] + pool[:offset]
            return eigenvalues, eigenvectors, order, selected, "band_channel_stratified"

    eigenvalues, eigenvectors = np.linalg.eigh(features.T @ features)
    order = [
        int(i)
        for i in np.argsort(eigenvalues)[::-1]
        if eigenvalues[i] > max(float(eigenvalues.max()) * 1e-10, 1e-20)
    ]
    if len(order) < 16:
        raise ValueError("at least sixteen observable readout directions required")
    pool = order[2:]
    offset = (6 * int(seed_or_round)) % len(pool)
    selected = order[:2] + pool[offset:] + pool[:offset]
    return eigenvalues, eigenvectors, order, selected, "global_pca"


def residual_paths(score):
    return score.get("residual_paths") or [score["residual_path"]]


def validate_paired_coordinates(coordinates, paired_radii):
    z = np.asarray(coordinates, np.float64)
    r = np.asarray(paired_radii, np.float64)
    if (
        r.ndim != 1
        or len(r) == 0
        or not np.all(np.isfinite(r))
        or not np.all(r > 0)
        or z.shape != (2 * len(r), len(r))
    ):
        raise ValueError("malformed antithetic paired_radii coordinate layout")
    expected = np.concatenate((np.diag(r), -np.diag(r)))
    if not np.allclose(z, expected, rtol=1e-7, atol=1e-12):
        raise ValueError("malformed antithetic paired_radii coordinate layout")
    return r


def residual_moments(center_paths, member_paths, coordinates, paired_radii=None):
    """Fit D=Z J^T, then solve (J^T J + lambda I)d=-J^T R_c in residual blocks."""
    z = np.asarray(coordinates, np.float64)
    r = validate_paired_coordinates(z, paired_radii) if paired_radii is not None else None
    if len(member_paths) != len(z):
        raise ValueError("member_paths count must match coordinate rows")
    eng = engine()
    if eng is not None:
        return eng.call(
            "residual",
            {
                "center_paths": center_paths,
                "member_paths": member_paths,
                "coordinates": coordinates,
                "paired_radii": None if r is None else r,
            },
        )
    pinv = np.linalg.pinv(z, rcond=1e-10)
    d = z.shape[1]
    gram = np.zeros((d, d), dtype=np.float64)
    cross = np.zeros(d, dtype=np.float64)
    pair_stats = np.zeros((5, d), dtype=np.float64) if r is not None else None
    count = 0
    for part, cp in enumerate(center_paths):
        center = load_array(cp, mmap_mode="r").reshape(-1)
        members = [load_array(paths[part], mmap_mode="r").reshape(-1) for paths in member_paths]
        if any(a.shape != center.shape for a in members):
            raise ValueError("unaligned public residuals")
        for start in range(0, len(center), 65536):
            c = np.asarray(center[start : start + 65536], np.float64)
            differences = np.stack(
                [np.asarray(a[start : start + 65536], np.float64) - c for a in members]
            )
            response = pinv @ differences
            gram += response @ response.T
            cross += response @ c
            if pair_stats is not None:
                diff_plus = differences[:d]
                diff_minus = differences[d:]
                odd = 0.5 * (diff_plus - diff_minus)
                even = 0.5 * (diff_plus + diff_minus)
                pair_stats[0] += np.sum(even * even, axis=1)
                pair_stats[1] += np.sum(odd * odd, axis=1)
                pair_stats[2] += np.sum(diff_plus * diff_plus, axis=1)
                pair_stats[3] += np.sum(diff_minus * diff_minus, axis=1)
                pair_stats[4] += np.sum(diff_plus * (-diff_minus), axis=1)
            count += len(c)
    gram /= count
    cross /= count
    meta = {
        "coordinate_rank": int(np.linalg.matrix_rank(z)),
        "residual_count": count,
        "coordinate_dimension": len(gram),
        "centered_on": "deterministic_policy",
    }
    if pair_stats is not None:
        e_diag = pair_stats[0] / count
        h_diag = pair_stats[1] / count
        plus_sq = pair_stats[2]
        minus_sq = pair_stats[3]
        anti_dot = pair_stats[4]
        meta.update(
            {
                "E_diag_fd": e_diag.tolist(),
                "H_diag_fd": h_diag.tolist(),
                "E_coordinate_convention": "dimensionless_probe_coordinates_z",
                "fd_even_over_odd_rms": np.sqrt(e_diag / np.maximum(h_diag, 1e-30)).tolist(),
                "fd_antithetic_cosine": (
                    anti_dot / np.sqrt(np.maximum(plus_sq * minus_sq, 1e-30))
                ).tolist(),
            }
        )
    return gram, cross, meta


def gn_steps(gram, cross, dampings=(0.001, 0.01, 0.05, 0.2)):
    scale = max(float(np.trace(gram) / len(gram)), 1e-12)
    return [
        (d, -np.linalg.solve(gram + d * scale * np.eye(len(gram)), cross), d * scale)
        for d in dampings
    ]


def residual_system(center_paths, member_paths, coordinates, damping=0.01):
    gram, cross, meta = residual_moments(center_paths, member_paths, coordinates)
    _, step, ridge = gn_steps(gram, cross, (damping,))[0]
    return step, {**meta, "ridge": ridge}


__all__ = [
    "_band_head_basis",
    "anchor_forward",
    "anchor_output",
    "body_candidates",
    "dense_body_candidates",
    "gn_steps",
    "head_map",
    "residual_moments",
    "residual_paths",
    "residual_system",
    "validate_paired_coordinates",
]
