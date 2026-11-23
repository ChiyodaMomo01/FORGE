"""Collect coherent probes, fit a response and screen projected updates."""

import numpy as np
import torch

from .common import good, save, set_model
from .projection import project
from .response_data import ResponseTrace
from .tangent.inverse import inverse_actions
from .tangent.model import SplitTangent
from .tangent.traces import IntervalTrace
from .tangent.validation import long_rollout_check, validate_response


def closed_loop_stage(s, k, folder, center, screen, trials):
    """Independent forcing identifies the active policy's closed-loop response.

    Unseen seeds decide whether the inverse/projection is allowed. Rejected
    response models retain all evidence and leave the accepted GN policy alone.
    """
    fitting_trials = trials[:8]
    base = s.score(
        center,
        trials=fitting_trials,
        trace=True,
        coherent=True,
        label=f"main-r{k}-closed-center",
    )
    attempts = []
    if not good(base):
        return center, {
            "status": "center_query_failed",
            "center": base,
            "attempts": attempts,
        }
    centers = {
        t.meta["trial"]: t
        for t in [IntervalTrace(ResponseTrace(path)) for path in base["trace_paths"]]
    }
    for attempt, sigma in enumerate((0.02, 0.01)):
        seed_base = 930000 + 1000 * k + 100 * attempt
        train = []
        held = []
        for role, seeds, destination in [
            ("fit", tuple(seed_base + 10 * j + j % 2 for j in range(10)), train),
            ("held", (seed_base + 97, seed_base + 98), held),
        ]:
            for seed in seeds:
                for sign in (-1, 1):
                    destination.append(
                        s.score(
                            center,
                            trials=fitting_trials,
                            sigma=sign * sigma,
                            seed=seed,
                            trace=True,
                            coherent=True,
                            label=f"main-r{k}-closed-{attempt}-{role}-{seed}-{sign}",
                        )
                    )
        row = {"sigma": sigma, "fit_collection": train, "held_collection": held}
        attempts.append(row)
        if not all(good(score) for score in [*train, *held]):
            row["status"] = "probe_query_failed"
            continue

        payload = {
            "theta": center,
            "centers": base["trace_paths"],
            "train": [path for score in train for path in score["trace_paths"]],
            "held": [path for score in held for path in score["trace_paths"]],
            "checkpoint": str(folder / f"closed-response-{attempt}.pt"),
            "sigma": sigma,
            "no_modulation": s.recipe.stage2_no_modulation,
            "one_step_only": s.recipe.stage2_one_step_only,
            "step_gain_limit": s.recipe.stage2_step_gain,
            "auto_step_gain": True,
        }
        from .distributed import engine

        execution = engine()
        corrections, fitted = (
            execution.call("response", payload)
            if execution is not None
            else fit_response_job(s.model, payload)
        )
        row.update(fitted)
        if row["status"] != "ready":
            save(folder / "closed-loop-progress.json", attempts)
            continue
        projected, projection = project(
            s.model,
            center,
            list(centers.values()),
            corrections,
            seed=9250 + k,
            l2_only=s.recipe.stage2_l2_projection,
            head_only=s.recipe.stage2_head_only,
        )
        candidates = []
        scores = []
        for alpha in (1.0, 0.5, 0.25):
            candidate = (center + alpha * (projected - center)).astype(np.float32)
            score = s.score(
                candidate,
                trials=trials,
                label=f"main-r{k}-projection-{attempt}-{alpha}",
            )
            beats = good(score) and score["mean_mse"] < screen["mean_mse"]
            scores.append({"alpha": alpha, "score": score, "beats_stage_I": beats})
            if beats:
                candidates.append((score["mean_mse"], candidate, alpha))
        accepted = bool(candidates)
        chosen = center
        if accepted:
            _, chosen, row["selected_alpha"] = min(candidates, key=lambda x: x[0])
        row.update(
            projection=projection,
            projection_screens=scores,
            status="accepted" if accepted else "true_screen_rejected",
        )
        save(folder / "closed-loop-progress.json", attempts)
        return chosen, {"status": row["status"], "center": base, "attempts": attempts}
    save(folder / "closed-loop-progress.json", attempts)
    return center, {"status": "stage_II_rejected", "center": base, "attempts": attempts}


def fit_response_job(model, payload):
    """Fit, validate and invert public traces, identically for every 2D workflow."""
    set_model(model, payload["theta"])
    centers = {
        trace.meta["trial"]: trace
        for trace in (IntervalTrace(ResponseTrace(path)) for path in payload["centers"])
    }

    def pairs(paths):
        return [
            (centers[trace.meta["trial"]], trace)
            for trace in (IntervalTrace(ResponseTrace(path)) for path in paths)
        ]

    tangent = SplitTangent(
        next(iter(centers.values())).width,
        model,
        next(model.parameters()).device,
        no_modulation=payload.get("no_modulation", False),
        one_step_only=payload.get("one_step_only", False),
        step_gain_limit=payload.get("step_gain_limit", 1.05),
        auto_step_gain=payload.get("auto_step_gain", True),
    )
    row = {"fit": tangent.fit(pairs(payload["train"]))}
    row["held_out"] = validate_response(tangent, pairs(payload["held"]))
    torch.save(
        {
            "schema": "public-interval-response-v1",
            "weights": tangent.weights.cpu(),
            "n": tangent.n,
            "context_mean": tangent.context_mean.cpu(),
            "context_scale": tangent.context_scale.cpu(),
            "state_channels": tangent.state_channels,
            "action_channels": tangent.action_channels,
            "no_modulation": tangent.no_modulation,
            "one_step_only": tangent.one_step_only,
            "step_gain": tangent.step_gain,
            "horizon_gain": tangent.horizon_gain,
            **row,
        },
        payload["checkpoint"],
    )
    if not row["held_out"]["passed"]:
        return None, {**row, "status": "held_out_rejected"}
    traces = list(centers.values())
    row["long_rollout"] = long_rollout_check(tangent, traces)
    if not row["long_rollout"]["passed"]:
        return None, {**row, "status": "long_rollout_rejected"}
    try:
        corrections, inverse = inverse_actions(tangent, traces, trust_rms=payload["sigma"])
    except FloatingPointError as error:
        return None, {**row, "status": "inverse_rejected", "inverse_error": str(error)}
    return corrections, {
        **row,
        "inverse": inverse,
        "status": "ready" if inverse["usable"] else "no_model_descent",
    }
