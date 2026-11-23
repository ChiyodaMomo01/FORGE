"""Exact per-shell operator norm bounds and affine step-gain projection."""

from __future__ import annotations

import torch


def _identity_weights(tangent, shape):
    w_id = torch.zeros(*shape, device=tangent.device, dtype=torch.float64)
    for c in range(tangent.state_channels):
        for s in range(tangent.shell_count):
            idx = 3 * s * tangent.input_channels + c
            if w_id.ndim == 2:
                w_id[:, idx] = 1.0
            else:
                w_id[:, idx, c] = 1.0
    return w_id


def _prior_weights(tangent, shape, use_increment=True):
    if not use_increment:
        return torch.zeros(*shape, device=tangent.device, dtype=torch.float64)
    w_prior = _identity_weights(tangent, shape)
    for s, sv in enumerate(tangent.shell_values):
        if sv >= 150:
            for c in range(tangent.state_channels):
                idx = 3 * s * tangent.input_channels + c
                if w_prior.ndim == 2:
                    w_prior[:, idx] = 0.7
                else:
                    w_prior[:, idx, c] = 0.7
    return w_prior


def _shell_state_norms(tangent, spec_state):
    """Exact per-(interval, shell) operator norm bound for state shell weights."""
    t_steps, s_count, _, p_out, _ = spec_state.shape
    b_shells = tangent.basis[: 3 * s_count].reshape(s_count, 3, tangent.n, tangent.n // 2 + 1)
    sym = torch.einsum("tsdio,sdhw->toihw", spec_state.to(b_shells), b_shells).abs().flatten(3)
    shell_idx = (
        tangent.shell.flatten()
        .to(device=spec_state.device, dtype=torch.long)[None, None, None, :]
        .expand_as(sym)
    )
    mod_all = sym.new_zeros((t_steps, p_out, p_out, s_count))
    mod_all.scatter_reduce_(3, shell_idx, sym, reduce="amax", include_self=True)
    if p_out == 1:
        return mod_all[:, 0, 0]
    if p_out == 2:
        g00 = mod_all[:, 0].square().sum(1)
        g11 = mod_all[:, 1].square().sum(1)
        g01 = (mod_all[:, 0] * mod_all[:, 1]).sum(1)
        disc = ((g00 - g11).square() + 4 * g01.square() + 1e-30).sqrt()
        return ((g00 + g11 + disc) / 2).clamp(min=1e-30).sqrt()
    return mod_all.square().sum((1, 2)).clamp(min=1e-30).sqrt()


def _clamp_step_gain(tangent, all_contexts, w_prior=None):
    """Affine active-only projection from verified feasible prior onto step_gain."""
    use_inc = not (tangent.no_modulation and tangent.one_step_only)
    if w_prior is None:
        w_prior = _prior_weights(tangent, tangent.weights.shape, use_increment=use_inc)
    w_ref = w_prior.to(dtype=tangent.weights.dtype, device=tangent.weights.device)
    is_2d = tangent.weights.ndim == 2
    act_mask = tangent.active if is_2d else tangent.active[:, :, None].expand_as(tangent.weights)
    tangent.weights = torch.where(act_mask, tangent.weights, w_ref)
    before_sb = tangent.bound_tensor(tangent.weights, all_contexts)[:, 0].detach().clone()

    p_out = tangent.state_channels
    in_ch = tangent.input_channels
    s_count = tangent.shell_count
    w3 = tangent.weights[:, :, None].clone() if is_2d else tangent.weights.clone()
    w_ref3 = w_ref[:, :, None] if w_ref.ndim == 2 else w_ref
    t_steps = w3.shape[0]
    budget = float(tangent.step_gain)

    b_ref = tangent.bound_tensor(w_ref3, all_contexts)[:, 0]
    if float(b_ref.max().item()) > budget:
        raise ValueError(
            f"prior state bound {float(b_ref.max().item()):.6g} exceeds step_gain {budget:.6g}"
        )

    if s_count > 0:
        spec_all = w3[:, : 3 * s_count * in_ch].reshape(t_steps, s_count, 3, in_ch, p_out)
        spec_ref = (
            w_ref3[:, : 3 * s_count * in_ch]
            .reshape(t_steps, s_count, 3, in_ch, p_out)[:, :, :, :p_out, :]
            .clone()
        )
        spec_cur = spec_all[:, :, :, :p_out, :].clone()
        d_spec = spec_cur - spec_ref
        norm_ref = _shell_state_norms(tangent, spec_ref)
        norm_cur = _shell_state_norms(tangent, spec_cur)
        sv_tensor = torch.tensor(tangent.shell_values, device=w3.device)
        hk_cap = min(1.0, budget)
        base_cap = torch.where(
            sv_tensor > 50,
            norm_cur.new_full((s_count,), hk_cap),
            norm_cur.new_full((s_count,), budget),
        )[None, :]
        shell_cap = torch.maximum(base_cap, norm_ref)
        need_s = norm_cur > shell_cap
        if bool(need_s.any()):
            lo_s = torch.zeros_like(norm_cur)
            hi_s = torch.ones_like(norm_cur)
            for _ in range(24):
                mid_s = 0.5 * (lo_s + hi_s)
                alpha_s = torch.where(need_s, mid_s, torch.ones_like(mid_s))
                cand_s = spec_ref + d_spec * alpha_s[:, :, None, None, None]
                ok_s = _shell_state_norms(tangent, cand_s) <= shell_cap
                lo_s = torch.where(ok_s, mid_s, lo_s)
                hi_s = torch.where(ok_s, hi_s, mid_s)
            alpha_s = torch.where(need_s, lo_s, torch.ones_like(lo_s))
            spec_all[:, :, :, :p_out, :] = spec_ref + d_spec * alpha_s[:, :, None, None, None]

    if not tangent.no_modulation:
        loc = w3[:, tangent.spectral_features :].reshape(
            t_steps, tangent.local_blocks, tangent.context_channels, in_ch, p_out
        )
        loc_state_backup = loc[:, :, :, :p_out, :].clone()
        loc[:, :, :, :p_out, :] = 0.0
        sb_spec_only = tangent.bound_tensor(w3, all_contexts)[:, 0]
        loc[:, :, :, :p_out, :] = loc_state_backup
        sb_full = tangent.bound_tensor(w3, all_contexts)[:, 0]
        need_loc = (sb_full > budget) & (sb_spec_only <= budget)
        if bool(need_loc.any()):
            lo_l = torch.zeros_like(sb_full)
            hi_l = torch.ones_like(sb_full)
            for _ in range(24):
                mid_l = 0.5 * (lo_l + hi_l)
                beta = torch.where(need_loc, mid_l, torch.ones_like(mid_l))
                loc[:, :, :, :p_out, :] = loc_state_backup * beta[:, None, None, None, None]
                ok_l = tangent.bound_tensor(w3, all_contexts)[:, 0] <= budget
                lo_l = torch.where(ok_l, mid_l, lo_l)
                hi_l = torch.where(ok_l, hi_l, mid_l)
            beta = torch.where(need_loc, lo_l, torch.ones_like(lo_l))
            loc[:, :, :, :p_out, :] = loc_state_backup * beta[:, None, None, None, None]

    state_ids = torch.tensor(tangent.state_ids, device=w3.device, dtype=torch.long)
    sb_now = tangent.bound_tensor(w3, all_contexts)[:, 0]
    need_global = sb_now > budget
    if bool(need_global.any()):
        state_ref = w_ref3[:, state_ids].clone()
        d_state = w3[:, state_ids].clone() - state_ref
        lo_g = torch.zeros_like(sb_now)
        hi_g = torch.ones_like(sb_now)
        for _ in range(28):
            mid_g = 0.5 * (lo_g + hi_g)
            gamma = torch.where(need_global, mid_g, torch.ones_like(mid_g))
            w3[:, state_ids] = state_ref + d_state * gamma[:, None, None]
            ok_g = tangent.bound_tensor(w3, all_contexts)[:, 0] <= budget
            lo_g = torch.where(ok_g, mid_g, lo_g)
            hi_g = torch.where(ok_g, hi_g, mid_g)
        gamma = torch.where(need_global, lo_g, torch.ones_like(lo_g))
        w3[:, state_ids] = state_ref + d_state * gamma[:, None, None]

    for _ in range(8):
        sb_chk = tangent.bound_tensor(w3, all_contexts)[:, 0]
        over = sb_chk > budget
        if not bool(over.any()):
            break
        state_ref = w_ref3[:, state_ids]
        d_cur = w3[:, state_ids] - state_ref
        shrink = torch.where(over, sb_chk.new_full(sb_chk.shape, 0.99), torch.ones_like(sb_chk))
        w3[:, state_ids] = state_ref + d_cur * shrink[:, None, None]
    else:
        sb_chk = tangent.bound_tensor(w3, all_contexts)[:, 0]
        over = sb_chk > budget
        if bool(over.any()):
            w3[:, state_ids] = torch.where(
                over[:, None, None], w_ref3[:, state_ids], w3[:, state_ids]
            )

    act3 = tangent.active[:, :, None].expand_as(w3)
    w3 = torch.where(act3, w3, w_ref3)
    tangent.weights = w3.squeeze(-1) if is_2d else w3
    tangent._operators = None
    after_sb = tangent.bound_tensor(tangent.weights, all_contexts)[:, 0].detach()
    if float(after_sb.max().item()) > budget * (1.0 + 1e-6):
        raise RuntimeError(
            f"step_gain clamp failed: {float(after_sb.max().item()):.6g} > {budget:.6g}"
        )
    return (after_sb / before_sb.clamp(min=1e-30)).clamp(max=1.0)


__all__ = [
    "_clamp_step_gain",
    "_identity_weights",
    "_prior_weights",
    "_shell_state_norms",
]
