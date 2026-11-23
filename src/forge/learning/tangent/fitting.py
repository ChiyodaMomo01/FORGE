"""Ridge identification and constrained rollout calibration from fit probes."""

from __future__ import annotations

import numpy as np
import torch

from .gain_clamp import (
    _clamp_step_gain,
    _identity_weights,
    _prior_weights,
    _shell_state_norms,
)
from .refine import refine_rollout
from .traces import output_difference


def fit_response(tangent, pairs, ridge=1e-6):
    if not pairs:
        raise ValueError("public interval pairs required")
    tangent.prepare_contexts([c for c, _ in pairs])
    horizon = pairs[0][0].steps
    f = tangent.features_count
    p_out = tangent.state_channels
    gram = torch.zeros(horizon, f, f, device=tangent.device, dtype=torch.float64)
    cross_shape = (
        (horizon, f) if (p_out == 1 and tangent.input_channels == 2) else (horizon, f, p_out)
    )
    cross = torch.zeros(*cross_shape, device=tangent.device, dtype=torch.float64)
    counts = torch.zeros(horizon, device=tangent.device, dtype=torch.float64)
    observed_growth = 1.0
    probe_ms = np.zeros(horizon, dtype=np.float64)
    probe_count = 0
    use_increment = not (tangent.no_modulation and tangent.one_step_only)
    tangent.fit_keys = set()
    for center, probe in pairs:
        delta = output_difference(center, probe)
        tangent.fit_keys.add((probe.meta["trial"], probe.meta["seed"]))
        norms = np.linalg.norm(delta[1:].reshape(horizon, -1), axis=1)
        probe_ms += norms * norms
        probe_count += 1
        if norms[0] > 1e-12:
            observed_growth = max(observed_growth, float(norms.max() / norms[0]))
        contexts = tangent.contexts(center)
        for t in range(horizon):
            dx = torch.as_tensor(delta[t], device=tangent.device)
            forcing = torch.as_tensor(probe.forcing[t] - center.forcing[t], device=tangent.device)
            target = torch.as_tensor(delta[t + 1], device=tangent.device)
            if use_increment:
                target = target - dx
            feature = tangent.features(contexts[t], dx, forcing)
            x = feature.permute(0, 2, 3, 1).reshape(-1, f).double()
            y = (
                target.flatten().double()
                if cross.ndim == 2
                else target.permute(0, 2, 3, 1).reshape(-1, p_out).double()
            )
            gram[t] += x.T @ x
            cross[t] += x.T @ y
            counts[t] += len(x)
    if bool((counts == 0).any()):
        raise ValueError("public interval pairs required")
    gram /= counts[:, None, None]
    if cross.ndim == 2:
        cross /= counts[:, None]
    else:
        cross /= counts[:, None, None]
    energy = gram.diagonal(dim1=-2, dim2=-1).sum(1) / f
    damping = (energy * ridge).clamp(min=1e-18)
    feat_idx = torch.arange(f, device=tangent.device)
    ch_idx = feat_idx % tangent.input_channels
    diag_gram = gram.diagonal(dim1=-2, dim2=-1)
    damping_vec = torch.zeros(horizon, f, device=tangent.device, dtype=torch.float64)
    active_mask = torch.zeros(horizon, f, device=tangent.device, dtype=torch.bool)
    shell_feat_end = 3 * tangent.shell_count * tangent.input_channels
    for ic in range(tangent.input_channels):
        spec_mask = (ch_idx == ic) & (feat_idx < shell_feat_end)
        loc_mask = (ch_idx == ic) & (feat_idx >= tangent.spectral_features)
        e_spec = diag_gram[:, spec_mask].mean(dim=1).clamp(min=1e-30)
        r_spec = (1e-3 if ic < p_out else max(ridge, 1e-5)) if use_increment else ridge
        damping_vec[:, spec_mask] = (e_spec[:, None] * r_spec).clamp(min=1e-18)
        active_mask[:, spec_mask] = diag_gram[:, spec_mask] > torch.maximum(
            e_spec[:, None] * 1e-12, e_spec.new_tensor(1e-24)
        )
        if bool(loc_mask.any()):
            if not tangent.no_modulation:
                e_loc = diag_gram[:, loc_mask].mean(dim=1).clamp(min=1e-30)
                r_loc = (2.5e-2 if ic < p_out else max(ridge, 5e-4)) if use_increment else ridge
                damping_vec[:, loc_mask] = (e_loc[:, None] * r_loc).clamp(min=1e-18)
                active_mask[:, loc_mask] = diag_gram[:, loc_mask] > torch.maximum(
                    e_loc[:, None] * 1e-12, e_loc.new_tensor(1e-24)
                )
            else:
                damping_vec[:, loc_mask] = 1e-18
    hk_forc_cols = [
        (3 * s + d) * tangent.input_channels + c
        for s, sv in enumerate(tangent.shell_values)
        if sv > 50
        for d in range(3)
        for c in range(p_out, tangent.input_channels)
    ]
    lk_forc_cols = [
        (3 * s + d) * tangent.input_channels + c
        for s, sv in enumerate(tangent.shell_values)
        if sv <= 50
        for d in range(3)
        for c in range(p_out, tangent.input_channels)
    ]
    if hk_forc_cols:
        hk_energy = float(diag_gram[:, hk_forc_cols].mean().item())
        lk_energy = float(diag_gram[:, lk_forc_cols].mean().item()) if lk_forc_cols else 0.0
        hk_excited = hk_energy > 1e-3 * max(lk_energy, 1e-30)
        if not hk_excited:
            active_mask[:, hk_forc_cols] = False
    else:
        hk_excited = True
    tangent.active = active_mask
    w_id = (
        _identity_weights(tangent, cross_shape)
        if use_increment
        else torch.zeros(*cross_shape, device=tangent.device, dtype=torch.float64)
    )
    w_prior = _prior_weights(tangent, cross_shape, use_increment=use_increment)
    hk_state_cols = [
        3 * s * tangent.input_channels + c
        for s, sv in enumerate(tangent.shell_values)
        if sv >= 150
        for c in range(p_out)
    ]
    tangent.weights = w_prior.clone()
    for t in range(horizon):
        active = tangent.active[t]
        size = int(active.sum())
        if size > 0:
            rhs_t = cross[t, active] + gram[t][active][:, active] @ (
                w_id[t, active] - w_prior[t, active]
            )
            d_t = damping_vec[t].clone()
            if use_increment and hk_state_cols:
                d_t[hk_state_cols] *= 10.0
            tangent.weights[t, active] = w_prior[t, active] + torch.linalg.solve(
                gram[t][active][:, active] + torch.diag(d_t[active]),
                rhs_t,
            )
    tangent.weights = tangent.weights.float()
    unique = list(dict.fromkeys(c for c, _ in pairs))
    all_contexts = torch.cat([tangent.contexts(c) for c in unique], dim=1)
    before = tangent.bound_tensor(tangent.weights, all_contexts)[:, 0]
    tangent.horizon_gain = min(16.0, max(2.0, 2 * observed_growth))
    tangent.horizon = horizon
    if not use_increment:
        tangent.step_gain = min(tangent.step_gain_limit, tangent.horizon_gain ** (1 / horizon))
        tangent.horizon_gain = tangent.step_gain**horizon
        factors = (tangent.step_gain / before.clamp(min=1e-30)).clamp(max=1)
        if float(factors.min().item()) < 1.0 - 1e-6:
            tangent.weights[:, tangent.state_ids] *= (
                factors[:, None] if tangent.weights.ndim == 2 else factors[:, None, None]
            )
            state_changed = True
        else:
            state_changed = False
    else:
        if (
            getattr(tangent, "auto_step_gain", False)
            and probe_count > 0
            and probe_ms[0] > 1e-30
            and horizon > 1
        ):
            probe_rms = np.sqrt(np.maximum(probe_ms, 1e-30))
            horizon_rms_gain = float(probe_rms.max() / probe_rms[0])
            tangent.step_gain = float(min(max(tangent.step_gain_limit, horizon_rms_gain), 10.5))
        else:
            tangent.step_gain = float(tangent.step_gain_limit)
        tangent.horizon_gain = min(float(tangent.step_gain) ** horizon, 1e100)
        w_pre_clamp = tangent.weights.detach().clone()
        factors = _clamp_step_gain(tangent, all_contexts, w_prior=w_prior)
        state_changed = not torch.allclose(tangent.weights, w_pre_clamp, atol=1e-7)
    if state_changed:
        for t in range(horizon):
            force_ids = torch.tensor(tangent.force_ids, device=tangent.device)[
                tangent.active[t, tangent.force_ids]
            ]
            active_state_ids = torch.tensor(
                [idx for idx in tangent.state_ids if bool(tangent.active[t, idx])],
                device=tangent.device,
                dtype=torch.long,
            )
            if len(force_ids) > 0:
                if len(active_state_ids) > 0:
                    delta_state = (
                        tangent.weights[t, active_state_ids].double() - w_id[t, active_state_ids]
                    )
                    rhs = (
                        cross[t, force_ids] - gram[t][force_ids][:, active_state_ids] @ delta_state
                    )
                else:
                    rhs = cross[t, force_ids]
                tangent.weights[t, force_ids] = torch.linalg.solve(
                    gram[t][force_ids][:, force_ids] + torch.diag(damping_vec[t, force_ids]),
                    rhs,
                ).float()
    refinement = (
        {"skipped": True, "reason": "one_step_only"}
        if tangent.one_step_only
        else refine_rollout(
            tangent,
            pairs,
            all_contexts,
            gram=gram,
            w_prior=w_prior,
            state_bound_before=before.detach() if use_increment else None,
        )
    )
    if use_increment and not tangent.one_step_only:
        _clamp_step_gain(tangent, all_contexts, w_prior=w_prior)
    if (
        use_increment
        and not hk_excited
        and getattr(tangent, "poly_count", 0) > 0
        and bool((tangent.poly_hk.abs() > 0).any())
    ):
        f0 = torch.cat(
            [torch.as_tensor(p.forcing[0] - c.forcing[0], device=tangent.device) for c, p in pairs],
            dim=0,
        )
        y1 = torch.cat(
            [torch.as_tensor(output_difference(c, p)[1], device=tangent.device) for c, p in pairs],
            dim=0,
        )
        freq_f0 = torch.fft.rfft2(f0, norm="ortho")
        feat_p0 = torch.fft.irfft2(
            freq_f0[:, None] * tangent.poly_full[None, :, None],
            s=(tangent.n, tangent.n),
            norm="ortho",
        ).flatten(1, 2)
        x0 = (
            feat_p0.permute(0, 2, 3, 1)
            .reshape(-1, tangent.poly_count * tangent.action_channels)
            .double()
        )
        y0 = y1.permute(0, 2, 3, 1).reshape(-1, p_out).double()
        gram0 = x0.T @ x0
        e0 = gram0.diagonal().mean().clamp(min=1e-30)
        damping0 = (e0 * max(ridge, 1e-6)).clamp(min=1e-18)
        w0_poly = torch.linalg.solve(
            gram0 + damping0 * torch.eye(x0.shape[1], device=tangent.device, dtype=torch.float64),
            x0.T @ y0,
        ).float()
        w3 = tangent.weights[:, :, None] if tangent.weights.ndim == 2 else tangent.weights
        poly_view = w3[:, shell_feat_end : tangent.spectral_features].reshape(
            horizon, tangent.poly_count, tangent.input_channels, p_out
        )
        poly_view[:, :, p_out:, :] = w0_poly.reshape(
            1, tangent.poly_count, tangent.action_channels, p_out
        )
    tangent._operators = None
    if not torch.isfinite(tangent.weights).all():
        raise FloatingPointError("nonfinite interval response")
    after, forcing_bound = tangent.norm_bounds(all_contexts)
    if use_increment and float(after) > float(tangent.step_gain) * (1.0 + 1e-6):
        raise RuntimeError(
            f"fitted state_norm_bound {float(after):.6g} exceeds step_gain {float(tangent.step_gain):.6g}"
        )
    return {
        "rows": int(counts.sum()),
        "no_modulation": tangent.no_modulation,
        "one_step_only": tangent.one_step_only,
        "features": f,
        "coefficient_count": int(tangent.weights.numel()),
        "squared_wavenumber_shells": tangent.shell_values,
        "active_features_by_interval": tangent.active.sum(1).cpu().tolist(),
        "ridge_relative": ridge,
        "ridge_absolute": damping.cpu().tolist(),
        "state_channels": tangent.state_channels,
        "action_channels": tangent.action_channels,
        "public_observation_channels": pairs[0][0].anchors.shape[2],
        "context_channels": tangent.context_channels,
        "context_scale": tangent.context_scale.cpu().tolist(),
        "context_mean": tangent.context_mean.cpu().tolist(),
        "context": "all public linear/symmetric quadratic channels and own G output; fit-center peak * sqrt(shell_count); no centering or output projection",
        "intervals": horizon,
        "rollout_refinement": refinement,
        "state_norm_before": float(before.max()),
        "state_norm_bound": after,
        "forcing_norm_bound": forcing_bound,
        "state_rescale": factors.cpu().tolist(),
        "step_gain_budget": tangent.step_gain,
        "window_gain_budget": tangent.horizon_gain,
        "fit_response_envelope": observed_growth,
        "fit_probe_keys": sorted(tangent.fit_keys),
        "zero_frequency": "included, response learned from unprojected public data",
        "initial_difference": "zero from same reset, no observation/output identity",
        "spectrum": "real gain and odd imaginary x/y phases; complex-modulus plus local infinity norm after exact identity-term regrouping",
        "feedback": "fixed-policy closed-loop interval response from public coherent forcing and raw field differences",
    }


__all__ = [
    "_clamp_step_gain",
    "_identity_weights",
    "_prior_weights",
    "_shell_state_norms",
    "fit_response",
    "refine_rollout",
]
