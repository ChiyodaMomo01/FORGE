"""Oracle request protocol validation and held-out Test ID freeze authorization."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..core.contracts import ACTIVE_SPLITS
from ..core.model_serde import array, digest, load_model
from ..schemas import is_valid_case_protocol


def build_handshake_info(
    model,
    bank,
    recipe,
    checkpoint_path: str | None,
    actor_numeric: dict,
) -> dict:
    if model.case_name != recipe.case_name:
        raise ValueError("cross-case checkpoint rejected")
    if not torch.equal(model.action_std, torch.ones_like(model.action_std)):
        raise ValueError("common unit action scale required")
    statistics = {
        "input_mean": model.network.input_mean.flatten()[: model.observation_channels]
        .cpu()
        .tolist(),
        "input_std": model.network.input_std.flatten()[: model.observation_channels].cpu().tolist(),
        "residual_rms": (model.residual_coefficient * model.output_scale).flatten().cpu().tolist(),
        "g_theta_scale": model.output_scale.flatten().cpu().tolist(),
    }
    theta = array(model)
    return {
        "theta": theta,
        "statistics": statistics,
        "parameters": len(theta),
        "n": model.network.config.nx,
        "width": model.network.config.width,
        "trials": {s: len(bank.maps[s]) for s in ACTIVE_SPLITS},
        "horizon": recipe.horizon,
        "group_size": 4,
        "window_ids": {
            s: [{"start": t, "parent_group": g} for t, g in bank.maps[s]] for s in ACTIVE_SPLITS
        },
        "initial_hash": digest(theta),
        "initialization": load_model(checkpoint_path, "cpu")[1].get("method")
        if checkpoint_path
        else "physics_start",
        "actor_numeric": actor_numeric,
    }


def verify_test_id_authorization(root: str | Path, recipe, theta) -> None:
    frozen = Path(root) / "research-freeze.json"
    if not frozen.exists():
        raise ValueError("Test ID denied before final research freeze")
    authorization = json.loads(frozen.read_text(encoding="utf-8"))
    if "case" in authorization and authorization.get("case") != recipe.case_name:
        raise ValueError("Test ID policy is not in the final frozen selection")
    if not is_valid_case_protocol(authorization.get("protocol"), recipe.case_name) or digest(
        theta
    ) not in authorization.get("allowed_policy_hashes", []):
        raise ValueError("Test ID policy is not in the final frozen selection")


def validate_rollout_request(
    request: dict, bank, root: str | Path, recipe
) -> tuple[str, list[int], str]:
    if request.get("op") != "rollout" or request.get("split") not in ACTIVE_SPLITS:
        raise ValueError("invalid public operation/split")
    split = request["split"]
    trials = (
        list(range(len(bank.maps[split]))) if request["trials"] is None else list(request["trials"])
    )
    if split == "test_id":
        verify_test_id_authorization(root, recipe, request["theta"])
    if (
        not trials
        or len(set(trials)) != len(trials)
        or any(t < 0 or t >= len(bank.maps[split]) for t in trials)
    ):
        raise ValueError("invalid canonical window trials")
    if split != "train" and (
        request["capacity"]
        or request["sigma"]
        or request.get("coherent")
        or request["residual"]
        or request.get("exploration", "correlated") != "correlated"
    ):
        raise ValueError("heldouts are score-only deterministic")
    exploration = request.get("exploration", "correlated")
    if exploration not in ("correlated", "independent"):
        raise ValueError("unsupported public action exploration law")
    if request.get("trace") and (
        split != "train" or request["mode"] != "network" or request.get("sealed")
    ):
        raise ValueError("continuous traces are train network only")
    return split, trials, exploration
