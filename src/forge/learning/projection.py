"""Sobolev regression of frozen action targets onto all network parameters."""

import math

import numpy as np
import torch

from .common import array, set_model


def sobolev_loss(prediction, target):
    delta = prediction - target
    loss = delta.square().mean()
    laplacian = torch.zeros_like(delta)
    for axis in (-2, -1):
        first = delta.roll(-1, axis) - delta
        second = delta.roll(-1, axis) - 2 * delta + delta.roll(1, axis)
        loss = loss + 0.1 * first.square().mean()
        laplacian = laplacian + second
    return loss + 0.01 * laplacian.square().mean()


def _sobolev_operator(field):
    out = field.clone()
    laplacian = torch.zeros_like(field)
    for axis in (-2, -1):
        second = field.roll(-1, axis) - 2.0 * field + field.roll(1, axis)
        out = out - 0.1 * second
        laplacian = laplacian + second
    biharmonic = torch.zeros_like(field)
    for axis in (-2, -1):
        biharmonic = biharmonic + (
            laplacian.roll(-1, axis) - 2.0 * laplacian + laplacian.roll(1, axis)
        )
    return out + 0.01 * biharmonic


def _build_eval_pairs(traces):
    pairs = []
    for i, tr in enumerate(traces):
        raw = tr.raw
        seen = set()
        for t in range(raw.steps):
            iv = int(raw.intervals[t])
            if iv not in seen:
                seen.add(iv)
                pairs.append((i, t))
    return pairs


def _batch_tensors(part, traces, corrections, device):
    observations = []
    targets = []
    for i, t in part:
        raw = traces[i].raw
        interval = int(raw.intervals[t])
        observations.append(torch.as_tensor(raw.observations[t], device=device))
        corr = torch.as_tensor(corrections[i][interval], device=device).detach()
        targets.append(torch.as_tensor(raw.actions[t], device=device) + corr)
    return torch.cat(observations), torch.cat(targets).detach()


@torch.no_grad()
def _evaluate_pairs(model, eval_pairs, traces, corrections, micro_pairs, loss_fn):
    device = next(model.parameters()).device
    total_loss = 0.0
    total_l2 = 0.0
    n_pairs = max(len(eval_pairs), 1)
    step = max(micro_pairs, 1)
    for offset in range(0, len(eval_pairs), step):
        part = eval_pairs[offset : offset + step]
        obs, tgt = _batch_tensors(part, traces, corrections, device)
        pred = model(obs)
        weight = len(part) / n_pairs
        total_loss += float(loss_fn(pred, tgt)) * weight
        total_l2 += float((pred - tgt).double().square().mean()) * weight
    return total_loss, math.sqrt(max(total_l2, 0.0))


@torch.no_grad()
def _solve_linear_head(
    model,
    start_w,
    start_b,
    eval_pairs,
    traces,
    corrections,
    micro_pairs,
    *,
    l2_only=False,
):
    device = next(model.parameters()).device
    head_mod = model.network.head
    has_bias = head_mod.bias is not None
    in_ch = head_mod.in_channels
    out_ch = head_mod.out_channels
    w_dim = in_ch + (1 if has_bias else 0)
    amp = float(getattr(model, "amplitude", 1.0))

    head_mod.weight.copy_(start_w)
    if has_bias:
        head_mod.bias.copy_(start_b)

    gram = torch.zeros((w_dim, w_dim), device=device, dtype=torch.float64)
    rhs = torch.zeros((w_dim, out_ch), device=device, dtype=torch.float64)
    captured = {}
    handle = head_mod.register_forward_pre_hook(
        lambda _, inp: captured.setdefault("feat", inp[0].detach())
    )
    step = max(micro_pairs, 1)
    total_pixels = 0.0
    try:
        for offset in range(0, len(eval_pairs), step):
            part = eval_pairs[offset : offset + step]
            obs, tgt = _batch_tensors(part, traces, corrections, device)
            captured.clear()
            pred0 = model(obs)
            raw_feat = captured["feat"]
            if has_bias:
                ones = torch.ones(
                    (raw_feat.shape[0], 1, raw_feat.shape[2], raw_feat.shape[3]),
                    device=device,
                    dtype=raw_feat.dtype,
                )
                phi = torch.cat([raw_feat, ones], dim=1).double() * amp
            else:
                phi = raw_feat.double() * amp
            op_phi = phi if l2_only else _sobolev_operator(phi)
            res = (tgt - pred0).double()
            phi_flat = phi.permute(1, 0, 2, 3).reshape(w_dim, -1)
            op_flat = op_phi.permute(1, 0, 2, 3).reshape(w_dim, -1)
            res_flat = res.permute(1, 0, 2, 3).reshape(out_ch, -1)
            gram += phi_flat @ op_flat.T
            rhs += op_flat @ res_flat.T
            total_pixels += float(phi_flat.shape[1])
    finally:
        handle.remove()

    norm_scale = max(total_pixels * max(out_ch, 1), 1.0)
    gram = 0.5 * (gram + gram.T) / norm_scale
    rhs = rhs / norm_scale
    diag_g = gram.diag().clamp(min=0.0)
    reg_vec = 1e-4 * diag_g.clone()
    feat_mean = float(diag_g[:in_ch].mean().item()) if in_ch > 0 else 0.0
    reg_vec[:in_ch] += max(1e-6 * feat_mean, 1e-18)
    if has_bias:
        reg_vec[in_ch:] += max(1e-6 * float(diag_g[in_ch:].mean().item()), 1e-18)
    dh = torch.linalg.solve(gram + torch.diag(reg_vec), rhs)
    head_mod.weight.copy_(start_w + dh[:in_ch].T.reshape_as(start_w).to(dtype=start_w.dtype))
    if has_bias:
        head_mod.bias.copy_(start_b + dh[in_ch:].reshape_as(start_b).to(dtype=start_b.dtype))


def project(
    model,
    theta,
    traces,
    corrections,
    updates=100,
    batch_pairs=16,
    micro_pairs=4,
    seed=9250,
    *,
    l2_only=False,
    head_only=False,
):
    """Variable-projection Sobolev regression onto linear readout and trunk parameters."""
    if not traces or len(traces) != len(corrections):
        raise ValueError("one frozen correction sequence per trace required")
    if updates < 0 or batch_pairs < 1 or micro_pairs < 1:
        raise ValueError("nonnegative updates and positive batch sizes required")
    device = next(model.parameters()).device
    set_model(model, theta)
    start = torch.as_tensor(theta, device=device)
    selected = list(model.network.head.parameters()) if head_only else list(model.parameters())
    loss_fn = (
        (lambda prediction, target: (prediction - target).square().mean())
        if l2_only
        else sobolev_loss
    )
    if updates == 0:
        return array(model), {
            "updates": 0,
            "loss": [],
            "parameters": len(theta),
            "batch_pairs": batch_pairs,
            "effective_parent_fields": traces[0].members * batch_pairs,
            "l2_only": bool(l2_only),
            "head_only": bool(head_only),
            "updated_parameters": sum(p.numel() for p in selected),
            "micro_pairs": micro_pairs,
            "graph": "frozen public observations and interval action targets; own network only",
        }

    eval_pairs = _build_eval_pairs(traces)
    start_w = model.network.head.weight.detach().clone()
    start_b = (
        model.network.head.bias.detach().clone() if model.network.head.bias is not None else None
    )
    loss_start, _ = _evaluate_pairs(model, eval_pairs, traces, corrections, micro_pairs, loss_fn)
    _solve_linear_head(
        model,
        start_w,
        start_b,
        eval_pairs,
        traces,
        corrections,
        micro_pairs,
        l2_only=l2_only,
    )
    loss_ho, rms_ho = _evaluate_pairs(model, eval_pairs, traces, corrections, micro_pairs, loss_fn)
    if loss_ho > loss_start:
        set_model(model, theta)
        loss_ho = loss_start
    if head_only:
        return array(model), {
            "updates": updates,
            "loss": [loss_start, loss_ho],
            "parameters": len(theta),
            "batch_pairs": batch_pairs,
            "effective_parent_fields": traces[0].members * batch_pairs,
            "l2_only": bool(l2_only),
            "head_only": True,
            "updated_parameters": sum(p.numel() for p in selected),
            "micro_pairs": micro_pairs,
            "graph": "frozen public observations and interval action targets; own network only",
        }

    ho_theta = array(model).copy()
    named_slices = list(
        zip(
            (name for name, _ in model.named_parameters()),
            parameter_slices(model),
            strict=True,
        )
    )
    body_slices = [(p, sl) for name, (p, sl) in named_slices if ".head." not in name]
    body_params = [p for p, _ in body_slices]
    head_params = list(model.network.head.parameters())
    body_denom = (
        sum(start[sl].square().sum() for _, sl in body_slices).clamp(min=1.0)
        if body_slices
        else start.new_tensor(1.0)
    )

    body_lr = 1e-4
    head_lr = 1e-4
    if body_params or head_params:
        cal_pairs = eval_pairs[: min(len(eval_pairs), batch_pairs)]
        model.zero_grad(set_to_none=True)
        for offset in range(0, len(cal_pairs), micro_pairs):
            part = cal_pairs[offset : offset + micro_pairs]
            obs, tgt = _batch_tensors(part, traces, corrections, device)
            (loss_fn(model(obs), tgt) * len(part) / len(cal_pairs)).backward()
        probe_pairs = cal_pairs[: min(len(cal_pairs), max(micro_pairs, 4))]
        obs_p, _ = _batch_tensors(probe_pairs, traces, corrections, device)
        eps = 1e-5
        with torch.no_grad():
            y_base = model(obs_p).double()
            if body_params:
                saved_body = [p.detach().clone() for p in body_params]
                for p in body_params:
                    if p.grad is not None:
                        adam_dir = p.grad / (p.grad.abs() + 1e-8)
                        p.add_(adam_dir, alpha=-eps)
                y_pert = model(obs_p).double()
                for p, sv in zip(body_params, saved_body, strict=True):
                    p.copy_(sv)
                s_body = float((y_pert - y_base).square().mean().sqrt().item()) / eps
                if s_body > 1e-12:
                    body_lr = float(np.clip((0.5 * rms_ho) / s_body, 1e-12, 1e-4))
            if head_params:
                saved_head = [p.detach().clone() for p in head_params]
                for p in head_params:
                    if p.grad is not None:
                        adam_dir = p.grad / (p.grad.abs() + 1e-8)
                        p.add_(adam_dir, alpha=-eps)
                y_pert_h = model(obs_p).double()
                for p, sv in zip(head_params, saved_head, strict=True):
                    p.copy_(sv)
                s_head = float((y_pert_h - y_base).square().mean().sqrt().item()) / eps
                if s_head > 1e-12:
                    head_lr = float(np.clip((0.25 * rms_ho) / s_head, 1e-12, 1e-3))
        model.zero_grad(set_to_none=True)

    optimizer = torch.optim.Adam(
        [
            {"params": body_params, "lr": body_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )
    t_max = max(updates, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(step, t_max) / t_max)),
    )
    rng = np.random.default_rng(seed)
    history = [loss_ho]
    for _ in range(updates):
        selections = [(int(rng.integers(len(traces))), None) for _ in range(batch_pairs)]
        selections = [(i, int(rng.integers(traces[i].raw.steps))) for i, _ in selections]
        model.zero_grad(set_to_none=True)
        total = 0.0
        for offset in range(0, batch_pairs, micro_pairs):
            part = selections[offset : offset + micro_pairs]
            obs, tgt = _batch_tensors(part, traces, corrections, device)
            loss = loss_fn(model(obs), tgt) * len(part) / batch_pairs
            loss.backward()
            total += float(loss.detach())
        if body_slices:
            trust = (
                sum((p.reshape(-1) - start[sl]).square().sum() for p, sl in body_slices)
                / body_denom
            )
            (0.001 * trust).backward()
            total += 0.001 * float(trust.detach())
        torch.nn.utils.clip_grad_norm_(selected, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        history.append(total)

    trained_theta = array(model).copy()
    best_theta = ho_theta.copy()
    best_loss = loss_ho
    for beta in (1.0, 0.5, 0.2, 0.05, 0.01, 2e-3, 4e-4):
        cand_theta = ho_theta + np.float32(beta) * (trained_theta - ho_theta)
        if not np.isfinite(cand_theta).all():
            continue
        set_model(model, cand_theta)
        direct_loss, _ = _evaluate_pairs(
            model, eval_pairs, traces, corrections, micro_pairs, loss_fn
        )
        best_beta_loss = direct_loss
        best_beta_arr = cand_theta.copy()
        try:
            _solve_linear_head(
                model,
                start_w,
                start_b,
                eval_pairs,
                traces,
                corrections,
                micro_pairs,
                l2_only=l2_only,
            )
            vp_loss, _ = _evaluate_pairs(
                model, eval_pairs, traces, corrections, micro_pairs, loss_fn
            )
            vp_arr = array(model).copy()
            if math.isfinite(vp_loss) and np.isfinite(vp_arr).all() and vp_loss < best_beta_loss:
                best_beta_loss = vp_loss
                best_beta_arr = vp_arr
        except (RuntimeError, FloatingPointError):
            pass
        if (
            math.isfinite(best_beta_loss)
            and np.isfinite(best_beta_arr).all()
            and best_beta_loss < best_loss
        ):
            best_loss = best_beta_loss
            best_theta = best_beta_arr
            break

    set_model(model, best_theta)
    history.append(best_loss)
    return array(model), {
        "updates": updates,
        "loss": history,
        "parameters": len(theta),
        "batch_pairs": batch_pairs,
        "effective_parent_fields": traces[0].members * batch_pairs,
        "l2_only": bool(l2_only),
        "head_only": False,
        "updated_parameters": sum(p.numel() for p in selected),
        "micro_pairs": micro_pairs,
        "graph": "frozen public observations and interval action targets; own network only",
    }


def parameter_slices(model):
    offset = 0
    for parameter in model.parameters():
        yield parameter, slice(offset, offset + parameter.numel())
        offset += parameter.numel()
