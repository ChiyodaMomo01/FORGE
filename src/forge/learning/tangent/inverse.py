"""Finite-window response, discrete adjoint and conjugate-gradient inversion."""

import numpy as np
import torch


class ClosedLoopResponse:
    def __init__(self, tangent, traces):
        self.tangent = tangent
        self.traces = traces
        self.steps = traces[0].steps
        if any(t.steps != self.steps for t in traces):
            raise ValueError("aligned output interval counts required")
        self.contexts = torch.cat([tangent.contexts(t) for t in traces], dim=1)
        state_channels = int(getattr(traces[0], "field_channels", tangent.state_channels))
        action_channels = int(getattr(traces[0], "action_channels", tangent.action_channels))
        self.state_shape = (
            self.steps,
            self.contexts.shape[1],
            state_channels,
            traces[0].height,
            traces[0].width,
        )
        self.action_shape = (
            self.steps,
            self.contexts.shape[1],
            action_channels,
            traces[0].height,
            traces[0].width,
        )

    @torch.no_grad()
    def forward(self, forcing):
        state = self.contexts.new_zeros(self.state_shape[1:])
        fields = []
        for t in range(self.steps):
            state = self.tangent.apply(self.contexts[t], state, forcing[t], t)
            fields.append(state)
        return torch.stack(fields)

    @torch.no_grad()
    def reverse(self, fields):
        adj = torch.zeros_like(fields[0])
        forcing = fields.new_zeros(self.action_shape)
        for t in range(self.steps - 1, -1, -1):
            adj, forcing[t] = self.tangent.adjoint(self.contexts[t], adj + fields[t], t)
        return forcing

    def gain_bound(self):
        """Finite response bound from the small matrix of causal block bounds."""
        bounds = self.tangent.bound_tensor(self.tangent.weights, self.contexts).detach()
        envelope = bounds.new_zeros((self.steps, self.steps))
        for k in range(self.steps):
            gain = bounds[k, 1]
            for t in range(k, self.steps):
                if t > k:
                    gain = gain * bounds[t, 0]
                envelope[t, k] = gain
        return float(torch.linalg.matrix_norm(envelope, ord=2))

    def residual(self):
        return torch.as_tensor(
            np.concatenate([t.task_residual() for t in self.traces], axis=1),
            device=self.tangent.device,
        )


def inverse_actions(tangent, traces, iterations=32, damping=0.01, trust_rms=0.02):
    response = ClosedLoopResponse(tangent, traces)
    residual = response.residual()
    s_res = -residual.clone()
    r = response.reverse(s_res)
    x = torch.zeros_like(r)
    p = r.clone()
    rr = float(r.double().square().sum())
    initial = rr
    best_x = x.clone()
    best_rr = rr
    best_obj = 0.5 * float(s_res.double().square().sum())
    max_norm_sq = float(x.numel()) * (float(trust_rms) ** 2)
    completed = 0
    boundary_hit = False
    for completed in range(1, iterations + 1):
        if rr <= 1e-20:
            completed -= 1
            break
        qp = response.forward(p)
        pp = float(p.double().square().sum())
        denominator = float(qp.double().square().sum()) + float(damping) * pp
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError("invalid interval CG curvature")
        alpha = rr / denominator
        x_next = x + alpha * p
        next_sq = float(x_next.double().square().sum())
        if next_sq >= max_norm_sq:
            xp = float((x.double() * p.double()).sum())
            xx = float(x.double().square().sum())
            disc = max(0.0, xp * xp + pp * max(0.0, max_norm_sq - xx))
            sigma = (-xp + float(np.sqrt(disc))) / max(pp, 1e-30)
            x_cand = x + sigma * p
            s_cand = s_res - sigma * qp
            r_cand = response.reverse(s_cand) - damping * x_cand
            rr_cand = float(r_cand.double().square().sum())
            obj_cand = 0.5 * float(s_cand.double().square().sum()) + 0.5 * float(damping) * float(
                x_cand.double().square().sum()
            )
            if not np.isfinite(rr_cand) or not np.isfinite(obj_cand):
                raise FloatingPointError("interval inverse diverged")
            if obj_cand < best_obj:
                best_obj = obj_cand
                best_x = x_cand
                best_rr = rr_cand
                boundary_hit = True
            break
        x = x_next
        s_res = s_res - alpha * qp
        r = response.reverse(s_res) - damping * x
        updated = float(r.double().square().sum())
        obj = 0.5 * float(s_res.double().square().sum()) + 0.5 * float(damping) * float(
            x.double().square().sum()
        )
        if not np.isfinite(updated) or not np.isfinite(obj):
            raise FloatingPointError("interval inverse diverged")
        if obj < best_obj:
            best_obj = obj
            best_x = x.clone()
            best_rr = updated
        else:
            # Stop if single-precision roundoff causes loss of A-conjugacy and raises the quadratic model.
            break
        if updated < max(initial * 1e-8, 1e-20):
            break
        p = r + (updated / rr) * p
        rr = updated
    x = best_x
    rr = best_rr
    if not torch.isfinite(x).all():
        raise FloatingPointError("interval inverse diverged")
    before = residual.square().mean()
    rms = float(np.sqrt(float(x.double().square().mean())))
    if rms > trust_rms * (1.0 + 1e-6):
        x = x * (trust_rms / max(rms, 1e-20))
    predicted = (residual + response.forward(x)).square().mean()
    usable = float(predicted) < float(before) and float(x.double().square().mean()) > 1e-20
    if not usable:
        x.zero_()
        predicted = before
    return list(x.split([t.members for t in traces], dim=1)), {
        "initial_normal_residual_squared": initial,
        "final_normal_residual_squared": float(rr),
        "action_rms_before_trust": float(rms),
        "trust_rms": trust_rms,
        "trust_boundary_hit": boundary_hit,
        "damping": damping,
        "CG_iterations": completed,
        "center_model_mse": float(before),
        "candidate_model_mse": float(predicted),
        "usable": usable,
        "control_intervals": response.steps,
    }
