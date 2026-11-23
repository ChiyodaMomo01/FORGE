"""Public-value ES and deterministic-center, coordinate-space EKI."""

from __future__ import annotations

import numpy as np

from .common import basis, good
from .subspace import (
    gn_steps,
    residual_moments,
    residual_paths,
    residual_system,
    validate_paired_coordinates,
)


def run_ensemble(s):
    r = s.recipe
    s.initialize()
    for k in range(r.fine_rounds):
        center = s.center.copy()
        trials = s.batch()
        u = basis(s.model, 3100 + k, 16)
        rng = np.random.default_rng(3200 + k)
        sigma = r.sigma_parameter
        z = rng.normal(size=(r.population // 2, 16)) * sigma
        z = (
            np.concatenate((z, -z))
            if s.method == "ES"
            else rng.normal(size=(r.population, 16)) * sigma
        )
        screen = s.score(
            center,
            trials=trials,
            label=f"r{k}-screen-center",
            residual=s.method == "EKI",
        )
        scores = [
            s.score(
                (center + u @ c).astype(np.float32),
                trials=trials,
                label=f"r{k}-member{j}",
                residual=s.method == "EKI",
            )
            for j, c in enumerate(z)
        ]
        losses = np.array([x["mean_mse"] for x in scores])
        candidates = []
        meta = {}
        screens = []
        if good(screen) and all(good(x) for x in scores):
            if s.method == "ES":
                direction = -(z.T @ (losses - losses.mean())) / (len(z) * sigma**2)
                direction /= max(float(np.linalg.norm(direction)), 1e-30)
                radii = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003)
            else:
                direction, meta = residual_system(
                    residual_paths(screen), [residual_paths(x) for x in scores], z
                )
                radii = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625)
            meta["coordinate_step_norm_before_linesearch"] = float(np.linalg.norm(direction))
            if np.linalg.norm(direction) > 0:
                for radius in radii:
                    theta = (center + u @ direction * radius).astype(np.float32)
                    sc = s.score(theta, trials=trials, label=f"r{k}-candidate-{radius}")
                    accepted = good(sc) and sc["mean_mse"] < screen["mean_mse"]
                    screens.append({"radius": radius, "score": sc, "beats_center": accepted})
                    if accepted:
                        candidates.append((sc["mean_mse"], theta, radius))
        accepted = bool(candidates)
        if accepted:
            _, s.center, chosen = min(candidates, key=lambda x: x[0])
            s.center = s.center.copy()
            meta["accepted_radius"] = chosen
        stop = s.validate(
            k + 1,
            {
                "sigma": sigma,
                "training_trials": trials,
                "member_scores": scores,
                "screen_center": screen,
                "candidate_scores": screens,
                "accepted": accepted,
                "acceptance_scope": "same batch only; full train once after selection",
                **meta,
            },
        )
        if stop:
            break
    return s.finish()


__all__ = [
    "gn_steps",
    "residual_moments",
    "residual_paths",
    "residual_system",
    "run_ensemble",
    "validate_paired_coordinates",
]
