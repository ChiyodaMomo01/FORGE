"""Constrained multi-step L-BFGS rollout calibration on fit probes."""

from __future__ import annotations

import torch

from .gain_clamp import _prior_weights
from .traces import output_difference


def refine_rollout(
    tangent,
    pairs,
    all_contexts,
    iterations=150,
    gram=None,
    w_prior=None,
    state_bound_before=None,
):
    unique_centers = list(dict.fromkeys(center for center, _ in pairs))
    ctx_list = []
    forcing_list = []
    target_list = []
    count = 0
    norm_sum = torch.zeros((), device=tangent.device, dtype=torch.float64)
    for center in unique_centers:
        c_forcing = []
        c_target = []
        for c_pair, probe in pairs:
            if c_pair is center:
                f_t = torch.as_tensor(probe.forcing - center.forcing, device=tangent.device)
                y_t = torch.as_tensor(output_difference(center, probe)[1:], device=tangent.device)
                c_forcing.append(f_t)
                c_target.append(y_t)
                count += y_t.numel()
                norm_sum = norm_sum + y_t.double().square().sum()
        ctx_list.append(tangent.contexts(center))
        forcing_list.append(torch.stack(c_forcing, dim=1))
        target_list.append(torch.stack(c_target, dim=1))
    contexts_all = torch.cat(ctx_list, dim=1)
    forcing_all = torch.cat(forcing_list, dim=2)
    target_all = torch.cat(target_list, dim=2)
    normalizer = (norm_sum / max(count, 1)).float().clamp(min=1e-20)
    gram_f = gram.float() if gram is not None else None
    w_ref = (
        w_prior.to(tangent.weights)
        if w_prior is not None
        else _prior_weights(
            tangent,
            tangent.weights.shape,
            use_increment=not (tangent.no_modulation and tangent.one_step_only),
        ).to(tangent.weights)
    )
    active = (
        tangent.active.to(tangent.weights)
        if tangent.weights.ndim == 2
        else tangent.active[:, :, None].to(tangent.weights)
    )
    s4 = tangent.local_ops.to(device=tangent.device)
    history = []
    stages = 2 if tangent.n >= 16 else 1
    baseline = tangent.weights.detach().clone()
    if state_bound_before is None and not (tangent.no_modulation and tangent.one_step_only):
        state_bound_before = tangent.bound_tensor(baseline, all_contexts)[:, 0].detach()
    gain_cap = (
        torch.minimum(
            (0.95 * state_bound_before.to(tangent.weights)).clamp(min=tangent.step_gain_limit),
            torch.full_like(
                state_bound_before,
                float(tangent.step_gain),
                dtype=tangent.weights.dtype,
            ),
        )
        if state_bound_before is not None
        else None
    )
    active_hk_freq_mask = None
    if gain_cap is not None:
        sv_t = torch.tensor(tangent.shell_values, device=tangent.device)
        act_shells = (
            tangent.active[:, : 3 * tangent.shell_count * tangent.input_channels]
            .reshape(tangent.horizon, tangent.shell_count, 3, tangent.input_channels)[
                :, :, :, : tangent.state_channels
            ]
            .any(dim=(0, 2, 3))
        )
        hk_shells = (sv_t > 50) & act_shells
        if bool(hk_shells.any()):
            active_hk_freq_mask = hk_shells[
                tangent.shell.to(device=tangent.device, dtype=torch.long)
            ]
    diff_base = (baseline - w_ref).abs()
    col_scale = diff_base.amax(dim=0, keepdim=True).clamp(min=1e-2)
    scale = col_scale.expand_as(baseline).clone()
    parameter = torch.nn.Parameter(torch.zeros_like(baseline))
    poly_cols = getattr(tangent, "poly_count", 0) * tangent.input_channels
    base_cols = max(baseline.shape[1] - poly_cols, 1)
    reg_scale = float(baseline.shape[1]) / float(base_cols)
    final = tangent.weights.new_zeros(())

    for _stage in range(stages):
        optimizer = torch.optim.LBFGS(
            [parameter],
            lr=1.0,
            max_iter=iterations,
            history_size=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )

        def constrained():
            return baseline + active * scale * parameter

        def objective(backward, opt=optimizer):
            if backward:
                opt.zero_grad(set_to_none=True)
            w_constrained = constrained()
            w_leaf = w_constrained.detach().requires_grad_(backward)
            w3 = w_leaf[:, :, None] if w_leaf.ndim == 2 else w_leaf
            spec = w3[:, : tangent.spectral_features].reshape(
                -1,
                tangent.spectral_count,
                tangent.input_channels,
                tangent.state_channels,
            )
            spectrum = torch.einsum("tkio,khw->toihw", spec.to(tangent.basis), tangent.basis)
            local = w3[:, tangent.spectral_features :].reshape(
                -1,
                tangent.local_blocks,
                tangent.context_channels,
                tangent.input_channels,
                tangent.state_channels,
            )
            if not tangent.no_modulation:
                mod_all = torch.einsum("tbjhw,tmjio->tmboihw", contexts_all, local.to(contexts_all))
            state = torch.zeros_like(target_all[0])
            values = []
            for t in range(tangent.horizon):
                v = torch.cat((state, forcing_all[t]), dim=2)
                freq = torch.fft.rfft2(v, norm="ortho")
                spec_val = torch.einsum("oihw,kbihw->kbohw", spectrum[t], freq)
                if not tangent.no_modulation:
                    m_t = mod_all[t]
                    op_v = torch.fft.irfft2(
                        s4[None, None, :, None] * freq[:, :, None],
                        s=(tangent.n, tangent.n),
                        norm="ortho",
                    )
                    spatial = torch.einsum("boihw,kbihw->kbohw", m_t[0], v) + torch.einsum(
                        "dboihw,kbdihw->kbohw", m_t[1:5], op_v
                    )
                    flux_out = torch.einsum("dboihw,kbihw->kbodhw", m_t[5:9], v)
                    f_flux = torch.fft.rfft2(flux_out, norm="ortho")
                    spec_val = spec_val + (f_flux * s4.to(f_flux)[None, None, None]).sum(3)
                    state = (
                        torch.fft.irfft2(spec_val, s=(tangent.n, tangent.n), norm="ortho") + spatial
                    )
                else:
                    state = torch.fft.irfft2(spec_val, s=(tangent.n, tangent.n), norm="ortho")
                values.append(state)
            pred_all = torch.stack(values)
            error = (pred_all - target_all).square().sum() / (count * normalizer)
            if gram_f is not None:
                b3 = baseline[:, :, None] if baseline.ndim == 2 else baseline
                dw = w3 - b3
                one_step_excess = torch.einsum("tfo,tfg,tgo->", dw, gram_f, dw) / (
                    tangent.horizon * tangent.state_channels * normalizer
                )
                error = error + 0.1 * one_step_excess
            if gain_cap is not None:
                sb_vec = tangent.bound_tensor(w3, contexts_all)[:, 0]
                bound_pen = 20.0 * ((torch.relu(sb_vec - gain_cap) / gain_cap).square().mean())
                error = error + bound_pen
                if active_hk_freq_mask is not None:
                    hk_cap = min(1.0, float(tangent.step_gain))
                    p_out = tangent.state_channels
                    hk_abs = spectrum[:, :p_out, :p_out][:, :, :, active_hk_freq_mask].abs()
                    if p_out == 1:
                        hk_norm = hk_abs[:, 0, 0]
                    elif p_out == 2:
                        g00 = hk_abs[:, 0].square().sum(1)
                        g11 = hk_abs[:, 1].square().sum(1)
                        g01 = (hk_abs[:, 0] * hk_abs[:, 1]).sum(1)
                        disc = ((g00 - g11).square() + 4 * g01.square() + 1e-30).sqrt()
                        hk_norm = ((g00 + g11 + disc) / 2).clamp(min=1e-30).sqrt()
                    else:
                        hk_norm = hk_abs.square().sum((1, 2)).clamp(min=1e-30).sqrt()
                    error = error + 20.0 * torch.relu(hk_norm - hk_cap).square().mean()
            if not torch.isfinite(error):
                raise FloatingPointError("nonfinite fitted-linear calibration")
            if backward:
                error.backward()
                if w_leaf.grad is not None:
                    w_constrained.backward(w_leaf.grad)
            tangent.weights = w_constrained.detach()
            tangent._operators = None
            regularizer = (5e-5 * reg_scale) * parameter.square().mean()
            if backward:
                regularizer.backward()
                history.append(float(error.detach()))
            return error.detach() + regularizer.detach()

        with torch.enable_grad():
            optimizer.step(lambda ob=objective: ob(True))
        with torch.no_grad():
            final = objective(False)
            tangent.weights = tangent.weights.detach().clone()
            tangent._operators = None

    return {
        "iterations_limit": iterations * stages,
        "closure_evaluations": len(history),
        "training_relative_squared_errors": history,
        "regularized_objective": float(final),
        "parameters": int(parameter.numel()),
        "source": "fit probes only; raw public differences; no HR labels or native derivatives",
    }


__all__ = ["refine_rollout"]
