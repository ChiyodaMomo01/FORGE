"""Own-network output geometry, rank-4 drift compensation, and Stage I trust bounds.

Every direction is measured on frozen public observations. Solver derivatives,
operators, and hidden states are never used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.func import jvp

from .common import indices
from .subspace import (
    _band_head_basis,
    anchor_forward,
    anchor_output,
    body_candidates,
    dense_body_candidates,
    head_map,
)

_HEAD_CAP_BACKTRACK_LIMIT = 24
_BODY_PROBE_REL_CAP = 0.01
_BODY_PROBE_ABS_FLOOR = 1e-2
_BODY_STEP_REL_CAP = 0.05
_BODY_STEP_ABS_FLOOR = 5e-2
_OWN_G_NL_RATIO_LIMIT = 0.25


@dataclass
class RoundSearchContext:
    model: object
    center: np.ndarray
    observations: np.ndarray
    tangent_dirs: np.ndarray
    radii: np.ndarray
    head_idx: np.ndarray
    body_idx: np.ndarray
    slices: list
    compensation_rank: int
    geometry: dict
    base_output: np.ndarray | None = None
    direction_responses: np.ndarray | None = None
    head_cols: tuple = tuple(range(8))
    body_cols: tuple = tuple(range(8, 16))

    def apply_map(self, coordinates):
        """Cancel nonlinear hidden-feature drift while preserving exact first-order Jacobian."""
        coord = np.asarray(coordinates, dtype=np.float64)
        delta = self.tangent_dirs @ coord
        candidate = (self.center + delta).astype(np.float32)
        if not self.compensation_rank or not np.any(delta[self.body_idx]):
            return candidate
        if self.base_output is not None and self.direction_responses is not None:
            intended = self.base_output + self.direction_responses @ coord
        else:
            head_only = self.center.copy()
            head_only[self.head_idx] = candidate[self.head_idx]
            intended = anchor_output(self.model, head_only, self.observations, self.slices)
        actual = anchor_output(self.model, candidate, self.observations, self.slices)
        features = head_map(self.model, candidate, self.observations)
        u, singular, vt = np.linalg.svd(features, full_matrices=False)
        keep = min(
            self.compensation_rank,
            int(np.sum(singular > max(singular[0] * 1e-10, 1e-20))),
        )
        if keep:
            correction = vt[:keep].T @ ((u[:, :keep].T @ (intended - actual)) / singular[:keep])
            candidate[self.head_idx] += correction.astype(np.float32)
        return candidate

    def cap_step(
        self,
        coordinates,
        mode="conditional_resolve",
        system=None,
        cross=None,
        gram=None,
        center_mse=None,
    ):
        """Cap each hidden tensor, re-solve head coordinates, and bound trajectory/head steps."""
        step = np.asarray(coordinates, dtype=np.float64).copy()
        delta = self.tangent_dirs @ step
        factor = 1.0
        for name, sl, _ in self.slices:
            if ".head." not in name:
                cap = _BODY_STEP_REL_CAP * max(
                    float(np.linalg.norm(self.center[sl])), _BODY_STEP_ABS_FLOOR
                )
                factor = min(factor, cap / max(float(np.linalg.norm(delta[sl])), 1e-30))
        h_cols = np.asarray(self.head_cols, dtype=int)
        b_cols = np.asarray(self.body_cols, dtype=int)
        if mode == "uniform_ray":
            step *= factor
        else:
            if len(b_cols):
                step[b_cols] *= factor
            if factor < 1 and mode == "conditional_resolve" and system is not None and len(h_cols):
                rhs = -cross[h_cols].copy()
                if len(b_cols):
                    rhs = rhs - system[np.ix_(h_cols, b_cols)] @ step[b_cols]
                step[h_cols] = np.linalg.solve(system[np.ix_(h_cols, h_cols)], rhs)
            delta_post = self.tangent_dirs @ step
            post_factor = 1.0
            for name, sl, _ in self.slices:
                if ".head." not in name:
                    cap = _BODY_STEP_REL_CAP * max(
                        float(np.linalg.norm(self.center[sl])), _BODY_STEP_ABS_FLOOR
                    )
                    post_factor = min(
                        post_factor,
                        cap / max(float(np.linalg.norm(delta_post[sl])), 1e-30),
                    )
            if post_factor < 1.0:
                step *= post_factor
                factor = min(factor, post_factor)
        delta_head = self.tangent_dirs[self.head_idx] @ step
        head_cap = max(2.0 * float(np.linalg.norm(self.center[self.head_idx])), 4.0)
        head_scale = min(1.0, head_cap / max(float(np.linalg.norm(delta_head)), 1e-30))
        step *= head_scale
        if (
            gram is not None
            and center_mse is not None
            and np.isfinite(center_mse)
            and center_mse > 0
        ):
            rms_err = float(np.sqrt(center_mse))
            pred_rms = float(np.sqrt(max(float(step @ gram @ step), 0.0)))
            traj_scale = min(1.0, (0.35 * rms_err) / max(pred_rms, 1e-30))
            coord_scale = 1.0
            step *= traj_scale
        else:
            traj_scale = min(1.0, 0.05 / max(float(np.linalg.norm(step)), 1e-30))
            coord_scale = 1.0
            step *= traj_scale
        comp_scale = 1.0
        backtrack_exhausted = False
        center_h = self.center[self.head_idx].astype(np.float64)
        has_lin_model = self.base_output is not None and self.direction_responses is not None
        last_nl_ratio = 0.0

        def _verify_candidate(step_vec):
            cand_v = self.apply_map(step_vec)
            if not np.isfinite(cand_v).all():
                return False, np.inf, np.inf
            disp_v = float(np.linalg.norm(cand_v[self.head_idx].astype(np.float64) - center_h))
            nl_r = 0.0
            if has_lin_model and len(b_cols) and np.any(step_vec[b_cols]):
                lin_delta = self.direction_responses @ step_vec
                lin_norm = float(np.linalg.norm(lin_delta))
                if lin_norm > 1e-12:
                    act_delta = (
                        anchor_output(self.model, cand_v, self.observations, self.slices)
                        - self.base_output
                    )
                    nl_r = float(np.linalg.norm(act_delta - lin_delta)) / lin_norm
            return (
                disp_v <= head_cap and nl_r <= _OWN_G_NL_RATIO_LIMIT,
                disp_v,
                nl_r,
            )

        if self.compensation_rank or (has_lin_model and len(b_cols) and np.any(step[b_cols])):
            verified = False
            for attempt in range(_HEAD_CAP_BACKTRACK_LIMIT):
                ok_v, disp, last_nl_ratio = _verify_candidate(step)
                if ok_v:
                    verified = True
                    break
                ratios = [0.75 if attempt == 0 else 0.5]
                if np.isfinite(disp) and disp > head_cap and disp > 0.0:
                    ratios.append((head_cap / disp) * (1.0 - 1e-4))
                if np.isfinite(last_nl_ratio) and last_nl_ratio > _OWN_G_NL_RATIO_LIMIT:
                    ratios.append((_OWN_G_NL_RATIO_LIMIT / last_nl_ratio) * 0.9)
                ratio = max(min(ratios), 1e-4)
                step *= ratio
                comp_scale *= ratio
            if not verified:
                ok_v, disp, last_nl_ratio = _verify_candidate(step)
                verified = ok_v
            if not verified:
                backtrack_exhausted = True
                step = np.zeros_like(step)
                comp_scale = 0.0
                last_nl_ratio = 0.0
                cand = self.apply_map(step)
                if not np.isfinite(cand).all() or (
                    float(np.linalg.norm(cand[self.head_idx].astype(np.float64) - center_h))
                    > head_cap
                ):
                    raise RuntimeError("center parameter state violates finiteness or head cap")
        else:
            cand = self.apply_map(step)
            if not np.isfinite(cand).all():
                backtrack_exhausted = True
                step = np.zeros_like(step)
                comp_scale = 0.0
        return step, {
            "body_scale": factor,
            "head_scale": head_scale * comp_scale,
            "traj_scale": traj_scale,
            "coord_scale": coord_scale,
            "nl_ratio": float(last_nl_ratio),
            "backtrack_exhausted": backtrack_exhausted,
            "mode": mode,
        }


def build_round_search_context(
    model,
    center,
    obs,
    seed_or_round=0,
    extra=None,
    body_mode="structured",
    experiment_seed=0,
    compensation_rank=0,
):
    del extra
    if body_mode not in ("structured", "dense"):
        raise ValueError("unsupported body_mode")
    if compensation_rank not in (0, 4):
        raise ValueError("compensation_rank must be 0 or 4")
    center = np.asarray(center, dtype=np.float32).copy()
    obs = np.asarray(obs, dtype=np.float32)
    head, body, slices = indices(model)
    features = head_map(model, center, obs)
    eigenvalues, eigenvectors, order, selected, head_basis_mode = _band_head_basis(
        model, features, seed_or_round
    )
    forward = anchor_forward(model, obs, slices)
    flat = torch.as_tensor(center, device=next(model.parameters()).device)
    with torch.no_grad():
        base_output = forward(flat).detach().cpu().double().numpy()

    def derivative(direction):
        _, response = jvp(forward, (flat,), (torch.as_tensor(direction, device=flat.device),))
        return response.detach().cpu().double().numpy()

    vectors, responses, labels = [], [], []

    def add(direction, label):
        d = direction.astype(np.float64)
        response = derivative(direction.astype(np.float32))
        for _ in range(2):
            for previous, value in zip(vectors, responses, strict=True):
                coefficient = float(value @ response)
                d -= coefficient * previous
                response -= coefficient * value
        norm = float(np.linalg.norm(response))
        if norm < 1e-9:
            return False
        vectors.append(d / norm)
        responses.append(response / norm)
        labels.append(label)
        return True

    def head_direction(pc):
        direction = np.zeros_like(center)
        direction[head] = eigenvectors[:, pc] / np.sqrt(eigenvalues[pc])
        return direction

    zero_head = not np.any(center[head])
    pure_head_inputs = []
    if not zero_head:
        current_head_dir = np.zeros_like(center)
        current_head_dir[head] = center[head]
        if add(current_head_dir, "head_current_policy"):
            pure_head_inputs.append((current_head_dir, "head_current_policy"))

    used_pcs = set()
    for pc in selected:
        if len(vectors) >= 8:
            break
        hd = head_direction(pc)
        lbl = f"head_PC_{pc}"
        if add(hd, lbl):
            used_pcs.add(pc)
            pure_head_inputs.append((hd, lbl))
    if len(vectors) < 8:
        raise ValueError("at least eight observable readout directions required")

    remaining_pcs = [pc for pc in selected if pc not in used_pcs]
    pc_cursor = 0
    chosen_body_inputs = []
    fallback_slots = 0

    for slot in range(8):
        stream = (
            dense_body_candidates(center, slices, int(seed_or_round), slot, experiment_seed)
            if body_mode == "dense"
            else body_candidates(center, slices, int(seed_or_round), slot)
        )
        added = False
        for direction, label in stream:
            norm = float(np.linalg.norm(direction[body]))
            if norm < 1e-20:
                continue
            direction /= norm
            if zero_head:
                if pc_cursor >= len(remaining_pcs):
                    raise ValueError("insufficient unused head modes for zero_head pairing")
                direction = direction + head_direction(remaining_pcs[pc_cursor])
            if add(direction, label):
                if zero_head:
                    pc_cursor += 1
                chosen_body_inputs.append((direction, label))
                added = True
                break
        if not added:
            fallback_slots += 1

    if fallback_slots > 0:
        vectors, responses, labels = [], [], []
        for hd, lbl in pure_head_inputs:
            add(hd, lbl)
        while len(vectors) < 8 + fallback_slots and pc_cursor < len(remaining_pcs):
            pc = remaining_pcs[pc_cursor]
            pc_cursor += 1
            add(head_direction(pc), f"head_PC_fallback_{pc}")
        for bd, lbl in chosen_body_inputs:
            add(bd, lbl)
        while len(vectors) < 16 and pc_cursor < len(remaining_pcs):
            pc = remaining_pcs[pc_cursor]
            pc_cursor += 1
            add(head_direction(pc), f"head_PC_fallback_{pc}")

    if len(vectors) < 16:
        raise ValueError("fewer than sixteen observable Stage I directions")

    directions = np.stack(vectors, axis=1).astype(np.float32)
    actual = np.stack([derivative(directions[:, i]) for i in range(16)], axis=1)
    for _ in range(3):
        gram = actual.T @ actual
        if np.max(np.abs(gram - np.eye(16))) < 5e-5:
            break
        triangular = np.linalg.cholesky(gram)
        directions = np.linalg.solve(triangular, directions.T).T.astype(np.float32)
        actual = np.stack([derivative(directions[:, i]) for i in range(16)], axis=1)

    head_cols = [i for i in range(16) if not np.any(directions[body, i])]
    body_cols = [i for i in range(16) if i not in head_cols]

    radii = np.full(16, 0.02, dtype=np.float64)
    head_probe_cap = max(0.5 * float(np.linalg.norm(center[head])), 1.0)
    for i in range(16):
        h_norm = float(np.linalg.norm(directions[head, i]))
        if h_norm > 0.0:
            radii[i] = min(radii[i], head_probe_cap / h_norm)
    for i in body_cols:
        for name, sl, _ in slices:
            if ".head." not in name:
                cap = _BODY_PROBE_REL_CAP * max(
                    float(np.linalg.norm(center[sl])), _BODY_PROBE_ABS_FLOOR
                )
                radii[i] = min(radii[i], cap / max(float(np.linalg.norm(directions[sl, i])), 1e-30))
    geometry = {
        "round_seed": int(seed_or_round),
        "labels": labels,
        "head_basis_mode": head_basis_mode,
        "head_directions": len(head_cols),
        "body_directions": len(body_cols),
        "head_column_indices": head_cols,
        "body_column_indices": body_cols,
        "rotating_PC_indices": (
            selected[1:7] if head_basis_mode == "band_channel_stratified" else selected[2:8]
        ),
        "eligible_PC_indices": order,
        "feature_eigenvalues": eigenvalues.tolist(),
        "function_gram": (actual.T @ actual).tolist(),
        "body_basis_mode": body_mode,
        "body_compensation_rank": compensation_rank,
        "zero_head": zero_head,
        "derivatives": "own G JVP only",
        "metric": "network action-coordinate L2 on frozen public observations",
    }
    ctx = RoundSearchContext(
        model,
        center,
        obs,
        directions,
        radii,
        head,
        body,
        slices,
        compensation_rank,
        geometry,
        base_output=base_output,
        direction_responses=actual,
        head_cols=tuple(head_cols),
        body_cols=tuple(body_cols),
    )
    if compensation_rank:
        center_h = center[head].astype(np.float64)
        center_b = center[body]

        def _probe_ok(col_idx, rad):
            if not np.isfinite(rad) or rad <= 0.0:
                return False, np.inf
            z_pos = np.zeros(16, dtype=np.float64)
            z_pos[col_idx] = rad
            c_pos = ctx.apply_map(z_pos)
            c_neg = ctx.apply_map(-z_pos)
            if not (np.isfinite(c_pos).all() and np.isfinite(c_neg).all()):
                return False, np.inf
            if not (np.any(c_pos[body] != center_b) and np.any(c_neg[body] != center_b)):
                return False, 0.0
            disp_val = max(
                float(np.linalg.norm(c_pos[head].astype(np.float64) - center_h)),
                float(np.linalg.norm(c_neg[head].astype(np.float64) - center_h)),
            )
            return disp_val <= head_probe_cap, disp_val

        for i in body_cols:
            verified = False
            for attempt in range(_HEAD_CAP_BACKTRACK_LIMIT):
                ok, disp = _probe_ok(i, float(ctx.radii[i]))
                if ok:
                    verified = True
                    break
                if disp == 0.0 and float(ctx.radii[i]) > 0.0:
                    break
                if np.isfinite(disp) and disp > 0.0:
                    ratio = min(
                        (head_probe_cap / disp) * (1.0 - 1e-4),
                        0.75 if attempt == 0 else 0.5,
                    )
                else:
                    ratio = 0.5
                ctx.radii[i] = float(ctx.radii[i]) * ratio
            if not verified:
                ok, _ = _probe_ok(i, float(ctx.radii[i]))
                verified = ok
            if not verified:
                raise RuntimeError(
                    f"failed to find verified positive compensated probe radius for column {i} within head_probe_cap={head_probe_cap:.6g}"
                )
    return ctx


def function_basis(
    model,
    center,
    obs,
    seed_or_round=0,
    extra=None,
    body_mode="structured",
    experiment_seed=0,
    compensation_rank=0,
):
    context = build_round_search_context(
        model,
        center,
        obs,
        seed_or_round,
        extra,
        body_mode,
        experiment_seed,
        compensation_rank,
    )
    return context.tangent_dirs, context.geometry


__all__ = [
    "RoundSearchContext",
    "_BODY_PROBE_ABS_FLOOR",
    "_BODY_PROBE_REL_CAP",
    "_BODY_STEP_ABS_FLOOR",
    "_BODY_STEP_REL_CAP",
    "_HEAD_CAP_BACKTRACK_LIMIT",
    "_OWN_G_NL_RATIO_LIMIT",
    "_band_head_basis",
    "anchor_forward",
    "anchor_output",
    "body_candidates",
    "build_round_search_context",
    "dense_body_candidates",
    "function_basis",
    "head_map",
]
