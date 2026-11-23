"""Shared 2D parameter-space warm-up and public output-response correction."""

import numpy as np

from .closed_loop import closed_loop_stage
from .common import good, set_model
from .geometry import build_round_search_context
from .subspace import residual_moments, residual_paths


def stage_one(s, k, center, screen, trials):
    """Paired forward probes and a damped, readout-compensated subspace solve."""
    recipe = s.recipe
    context = build_round_search_context(
        s.model,
        center,
        s.obs,
        k,
        body_mode=recipe.body_basis_mode,
        experiment_seed=recipe.seed,
        compensation_rank=recipe.body_compensation_rank,
    )
    radii = context.radii
    coordinates = np.concatenate((np.diag(radii), -np.diag(radii)))
    members = [
        s.score(
            context.apply_map(z),
            trials=trials,
            residual=True,
            label=f"main-r{k}-paired-{j}",
        )
        for j, z in enumerate(coordinates)
    ]
    if not all(good(score) for score in members):
        raise RuntimeError("public GN difference query failed; evidence retained")
    gram, cross, moments = residual_moments(
        residual_paths(screen),
        [residual_paths(score) for score in members],
        coordinates,
        paired_radii=radii,
    )
    scale = max(float(np.trace(gram) / len(gram)), 1e-12)
    if recipe.damping_mode == "response_scaled":
        even = np.asarray(moments["E_diag_fd"]) / (np.maximum(radii, 1e-3) ** 2)
        diagonal = np.maximum(np.maximum(np.diag(gram), even), scale * 1e-8)
    else:
        diagonal = np.full(len(gram), scale)
    head_weight_slices = [
        np.arange(sl.start, sl.stop, dtype=int)
        for name, sl, _ in context.slices
        if name == "head.weight" or name.endswith(".head.weight")
    ]
    pen_idx = np.concatenate(head_weight_slices) if head_weight_slices else context.head_idx
    directions = context.tangent_dirs[pen_idx].astype(np.float64)
    pen_gram = directions.T @ directions
    pen_grad = directions.T @ center[pen_idx].astype(np.float64)
    penalty = float(recipe.head_weight_penalty_eta)
    candidates = [(screen["mean_mse"], center, screen, None)]
    screens = []
    for damping in (0.001, 0.01, 0.05, 0.2):
        diag_term = damping * np.diag(diagonal)
        sys0 = gram + diag_term
        step0, trust0 = context.cap_step(
            -np.linalg.solve(sys0, cross),
            recipe.stage_one_trust_mode,
            sys0,
            cross,
            gram=gram,
            center_mse=screen["mean_mse"],
        )
        pred_drop0 = -float(cross @ step0 + 0.5 * (step0 @ gram @ step0))
        step, trust, effective_eta, pred_drop = step0, trust0, 0.0, pred_drop0
        if penalty > 0.0 and pred_drop0 > 0.0:
            eta_ladder = [
                penalty,
                *[v for v in (3e-9, 1e-9, 3e-10, 1e-10) if v < penalty],
            ]
            for eta_try in eta_ladder:
                sys_try = gram + eta_try * pen_gram + diag_term
                grad_try = cross + eta_try * pen_grad
                step_try, trust_try = context.cap_step(
                    -np.linalg.solve(sys_try, grad_try),
                    recipe.stage_one_trust_mode,
                    sys_try,
                    grad_try,
                    gram=gram,
                    center_mse=screen["mean_mse"],
                )
                drop_try = -float(cross @ step_try + 0.5 * (step_try @ gram @ step_try))
                if drop_try >= 0.90 * pred_drop0:
                    step, trust, effective_eta, pred_drop = (
                        step_try,
                        trust_try,
                        float(eta_try),
                        drop_try,
                    )
                    break
        candidate = context.apply_map(step)
        if not np.isfinite(candidate).all():
            screens.append({"damping": damping, "status": "nonfinite_proposal"})
            continue
        score = s.score(candidate, trials=trials, label=f"main-r{k}-GN-{damping}")
        accepted = good(score) and score["mean_mse"] < screen["mean_mse"]
        screens.append(
            {
                "damping": damping,
                "effective_head_penalty_eta": effective_eta,
                "predicted_mse_reduction": pred_drop,
                "coordinates": step.tolist(),
                "own_g_trust": trust,
                "score": score,
                "beats_center": accepted,
            }
        )
        if accepted:
            candidates.append((score["mean_mse"], candidate, score, damping))
    _, chosen, chosen_score, damping = min(candidates, key=lambda value: value[0])
    return (
        chosen,
        chosen_score,
        {
            "function_geometry": context.geometry,
            "GN": moments,
            "members": members,
            "line_search": screens,
            "stage_I_damping": damping,
            "finite_difference_radii": radii.tolist(),
        },
    )


def run(s, resume_from=None):
    if resume_from is None:
        s.initialize()
        start = 0
    else:
        start, _ = s.resume_from_attempt(resume_from)
        if s.stopped:
            return s.finish()
    for k in range(start, s.recipe.fine_rounds):
        center = s.center.copy()
        trials = s.batch()
        folder = s.folder / f"round-{k + 1}"
        folder.mkdir()
        screen = s.score(
            center,
            trials=trials,
            residual=not s.recipe.disable_stage_one,
            label=f"main-r{k}-center",
        )
        if not good(screen):
            raise RuntimeError("public center query failed; evidence retained")
        set_model(s.model, center)
        if s.recipe.disable_stage_one:
            warmed, warmed_score = center, screen
            row = {"stage_I": {"status": "disabled", "reason": "stage_II_only_ablation"}}
        else:
            warmed, warmed_score, row = stage_one(s, k, center, screen, trials)
        if s.recipe.disable_stage_two:
            chosen = warmed
            stage_two = {"status": "disabled", "reason": "stage_I_only_ablation"}
        else:
            chosen, stage_two = closed_loop_stage(s, k, folder, warmed, warmed_score, trials)
        s.center = chosen.copy()
        set_model(s.model, s.center)
        stop = s.validate(
            k + 1,
            {
                **row,
                "center_score": screen,
                "stage_I_score": warmed_score,
                "stage_II": stage_two,
                "accepted": not np.array_equal(center, chosen),
            },
        )
        if stop:
            break
    return s.finish()
