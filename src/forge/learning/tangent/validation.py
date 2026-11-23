"""Held-out public response metrics and finite-window gain checks."""

import numpy as np
import torch

from .inverse import ClosedLoopResponse
from .traces import output_difference


def validate_response(tangent, pairs, relative_limit=0.5, cosine_limit=0.8, r2_limit=0.5):
    totals = {name: np.zeros(6, np.float64) for name in ("one_step", "rollout")}
    rows = []

    def measure(name, pred, target):
        p = pred.double().flatten()
        y = target.double().flatten()
        values = np.array(
            [
                float((p - y).square().sum()),
                float(y.square().sum()),
                float((p * y).sum()),
                float(p.square().sum()),
                float(y.sum()),
                len(y),
            ]
        )
        totals[name] += values
        return values

    with torch.no_grad():
        for center, probe in pairs:
            delta = output_difference(center, probe)
            if (probe.meta["trial"], probe.meta["seed"]) in tangent.fit_keys:
                raise ValueError("held-out action seed overlaps response fitting")
            contexts = tangent.contexts(center)
            for t in range(center.steps):
                pred = tangent.apply(
                    contexts[t],
                    torch.as_tensor(delta[t], device=tangent.device),
                    torch.as_tensor(probe.forcing[t] - center.forcing[t], device=tangent.device),
                    t,
                )
                measure(
                    "one_step",
                    pred,
                    torch.as_tensor(delta[t + 1], device=tangent.device),
                )
            response = ClosedLoopResponse(tangent, [center])
            forcing = torch.as_tensor(probe.forcing - center.forcing, device=tangent.device)
            actual = torch.as_tensor(delta[1:], device=tangent.device)
            pred = response.forward(forcing)
            values = measure("rollout", pred, actual)
            bound = tangent.norm_bounds(response.contexts)[0]
            rows.append(
                {
                    "trial": probe.meta["trial"],
                    "seed": probe.meta["seed"],
                    "sigma": probe.meta["sigma"],
                    "rollout_relative_error": float(np.sqrt(values[0] / max(values[1], 1e-30))),
                    "state_norm_bound": bound,
                    "response_norm_bound": response.gain_bound(),
                    "maximum_spatial_mean_error": float((pred - actual).mean((-2, -1)).abs().max()),
                }
            )
    metrics = {
        name: {
            "relative_error": float(np.sqrt(e / max(s, 1e-30))),
            "cosine": float(d / np.sqrt(max(s * p, 1e-30))),
            "r2": float(1 - e / max(s - total * total / max(count, 1), 1e-30)),
            "signal_squared": float(s),
            "raw_target_mean": float(total / max(count, 1)),
        }
        for name, (e, s, d, p, total, count) in totals.items()
    }
    passed = bool(rows) and all(
        np.isfinite(m["relative_error"])
        and m["signal_squared"] > 1e-20
        and m["relative_error"] <= relative_limit
        and m["r2"] >= r2_limit
        for m in metrics.values()
    )
    passed = (
        passed
        and metrics["rollout"]["cosine"] >= cosine_limit
        and all(
            r["rollout_relative_error"] <= 1
            and r["state_norm_bound"] <= tangent.step_gain * (1 + 1e-5)
            for r in rows
        )
    )
    return {
        "passed": passed,
        "metrics": metrics,
        "probes": rows,
        "relative_limit": relative_limit,
        "rollout_cosine_limit": cosine_limit,
        "r2_limit": r2_limit,
        "scope": "unseen TRAIN coherent forcing; raw unprojected public output differences; no HR labels; spatial means retained",
    }


def long_rollout_check(tangent, traces, steps=100):
    """Cycle frozen contexts and check only the fitted full-operator gain bound."""
    response = ClosedLoopResponse(tangent, traces)
    bound = tangent.norm_bounds(response.contexts)[0]
    state = torch.ones(response.state_shape[1:], device=tangent.device).normal_(
        generator=torch.Generator(device=tangent.device).manual_seed(888)
    )
    zero_forcing = torch.zeros(response.action_shape[1:], device=tangent.device)
    initial = float(torch.linalg.vector_norm(state))
    peak = initial
    identified = list(range(1, response.steps)) or [0]
    with torch.no_grad():
        for index in range(steps):
            t = identified[index % len(identified)]
            state = tangent.apply(response.contexts[t], state, zero_forcing, t)
            if not torch.isfinite(state).all():
                raise FloatingPointError("nonfinite long interval rollout")
            peak = max(peak, float(torch.linalg.vector_norm(state.double())))
    allowed = initial * max(1.0, bound) ** steps
    return {
        "passed": bool(np.isfinite(peak) and peak <= allowed * (1 + 1e-4)),
        "interval_steps": steps,
        "equivalent_callbacks": steps * int(traces[0].raw.steps / traces[0].steps),
        "state_norm_bound": bound,
        "observed_peak_norm": peak,
        "initial_norm": initial,
        "allowed_peak_norm": allowed,
        "cycled_intervals": identified,
        "check": "full-map finite-horizon gain; unexcited initial transition omitted from stress; no mass condition",
    }
