"""Public window contract, authoritative 2D benchmark recipes, and numeric execution contract."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..schemas import (
    ALL_2D_CASES,
    CANONICAL_CASE_PROTOCOLS,
    validate_case_protocol,
)

ACTIVE_SPLITS = ("train", "validation", "test_id")
METHODS = ("ES", "EKI", "PPO_global", "PPO_local", "emulator", "closure_rl", "main")

ALLOWED_PROFILE_TOP_KEYS: frozenset[str] = frozenset(
    {
        "case_name",
        "case",
        "benchmark_metadata",
        "recipe_overrides",
        "cli_defaults",
        "default_method",
        "output_root",
        "pairs_root",
        "original_root",
        "reference_root",
        "device",
        "source_sha256",
    }
)

ALLOWED_BENCHMARK_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "case_name",
        "default_initialization_workflow",
        "supported_workflows",
        "grid_coarsening",
        "closure_parameters",
        "window_horizon",
        "sample_dt",
        "fine_rounds",
        "directions",
        "training_groups_per_round",
        "window_budget_anchor",
    }
)

ALLOWED_CLI_DEFAULTS_KEYS: frozenset[str] = frozenset(
    {
        "--body-basis-mode",
        "--body-compensation-rank",
        "--stage-one-trust-mode",
        "--damping-mode",
        "--head-weight-penalty-eta",
        "--stage-one-only",
        "--stage-two-only",
        "--stage-two-no-modulation",
        "--stage-two-one-step-only",
        "--projection-l2-only",
        "--projection-head-only",
        "--max-rounds",
        "--target-window-budget",
        "--folder-name",
        "--ppo-lr",
        "--ad-lr",
        "--ad-epochs",
        "--ad-tag",
    }
)

SUPPORTED_WORKFLOWS: frozenset[str] = frozenset({"paired_pretrain", "physics_start"})

_EXPECTED_CASE_CLOSURE_PARAMETERS: dict[str, int] = {
    "cahn_hilliard": 489057,
    "gray_scott": 489057,
    "kolmogorov_ns": 490027,
    "mhd2d": 490057,
}


def _is_strict_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _is_strict_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, np.integer, np.floating))


@dataclass(frozen=True)
class Recipe:
    case_name: str = "cahn_hilliard"
    protocol: str = CANONICAL_CASE_PROTOCOLS["cahn_hilliard"]
    defer_test_id: bool = True
    shuffle_regression_halves: bool = True
    ad_patience: int = 4
    ad_plateau_relative: float = 0.002
    strict_actor_fp32: bool = True
    frames: int = 500
    stride: int = 10
    horizon: int = 20
    group_size: int = 4
    sample_dt: float = 0.1
    pretrain_epochs: int = 64
    pretrain_patience: int = 6
    ad_epochs: int = 128
    gradient_batch_groups: int = 2
    pretrain_lr: float = 1e-4
    ad_lr: float = 3e-5
    fine_rounds: int = 100
    patience: int = 5
    plateau_relative: float = 0.002
    training_groups_per_round: int = 40
    directions: int = 16
    population: int = 24
    sigma_parameter: float = 0.001
    sigma_action: float = 0.05
    ppo_epochs: int = 4
    ppo_lr: float = 1e-4
    closure_rl_lr: float = 1e-5
    emulator_lr: float = 3e-4
    emulator_updates: int = 60
    emulator_warmup_updates: int = 120
    emulator_policy_updates: int = 20
    emulator_policy_head_lr: float = 1e-4
    emulator_policy_body_lr: float = 3e-5
    window_budget_anchor: int = 701440
    validation_every_rounds: int = 2
    seed: int = 0
    actor_fp64: bool = False
    body_basis_mode: str = "dense"
    body_compensation_rank: int = 4
    stage_one_trust_mode: str = "conditional_resolve"
    damping_mode: str = "response_scaled"
    head_weight_penalty_eta: float = 0.0

    disable_stage_one: bool = False
    disable_stage_two: bool = False
    stage2_no_modulation: bool = False
    stage2_one_step_only: bool = False
    stage2_l2_projection: bool = False
    stage2_head_only: bool = False
    stage2_step_gain: float = 1.05
    ignore_early_stop: bool = False
    target_window_budget: int = 0

    def __post_init__(self):
        if (
            not isinstance(self.case_name, str)
            or self.case_name not in ALL_2D_CASES
            or not _is_strict_int(self.group_size)
            or self.group_size != 4
        ):
            raise ValueError("only supported 2D four-parent windows allowed")
        validate_case_protocol(self.protocol, self.case_name)

        bool_fields = (
            "defer_test_id",
            "shuffle_regression_halves",
            "strict_actor_fp32",
            "actor_fp64",
            "disable_stage_one",
            "disable_stage_two",
            "stage2_no_modulation",
            "stage2_one_step_only",
            "stage2_l2_projection",
            "stage2_head_only",
            "ignore_early_stop",
        )
        for name in bool_fields:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

        positive_int_fields = (
            "frames",
            "stride",
            "horizon",
            "pretrain_epochs",
            "pretrain_patience",
            "ad_epochs",
            "gradient_batch_groups",
            "fine_rounds",
            "patience",
            "training_groups_per_round",
            "directions",
            "population",
            "ppo_epochs",
            "emulator_updates",
            "emulator_warmup_updates",
            "emulator_policy_updates",
            "validation_every_rounds",
        )
        for name in positive_int_fields:
            val = getattr(self, name)
            if not _is_strict_int(val) or int(val) < 1:
                raise ValueError(f"{name} must be a positive integer")

        nonneg_int_fields = (
            "ad_patience",
            "window_budget_anchor",
            "target_window_budget",
        )
        for name in nonneg_int_fields:
            val = getattr(self, name)
            if not _is_strict_int(val) or int(val) < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if not _is_strict_int(self.seed) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if int(self.population) < 2:
            raise ValueError("population must be >= 2")

        positive_float_fields = (
            "sample_dt",
            "pretrain_lr",
            "ad_lr",
            "sigma_parameter",
            "sigma_action",
            "ppo_lr",
            "closure_rl_lr",
            "emulator_lr",
            "emulator_policy_head_lr",
            "emulator_policy_body_lr",
            "stage2_step_gain",
        )
        for name in positive_float_fields:
            val = getattr(self, name)
            if not _is_strict_real(val) or not math.isfinite(float(val)) or float(val) <= 0.0:
                raise ValueError(f"positive finite {name} required")

        nonneg_float_fields = (
            "ad_plateau_relative",
            "plateau_relative",
            "head_weight_penalty_eta",
        )
        for name in nonneg_float_fields:
            val = getattr(self, name)
            if not _is_strict_real(val) or not math.isfinite(float(val)) or float(val) < 0.0:
                raise ValueError(f"nonnegative finite {name} required")

        if self.frames < self.horizon:
            raise ValueError("positive full windows required")
        if self.body_basis_mode not in ("structured", "dense"):
            raise ValueError("unsupported body_basis_mode")
        if not _is_strict_int(self.body_compensation_rank) or self.body_compensation_rank not in (
            0,
            4,
        ):
            raise ValueError("unsupported body_compensation_rank; only 0 or 4 supported")
        if self.body_compensation_rank > 0 and (
            self.body_basis_mode != "dense" or self.damping_mode != "response_scaled"
        ):
            raise ValueError(
                "body_compensation_rank > 0 requires dense body_basis_mode, and response_scaled damping_mode"
            )
        if self.stage_one_trust_mode not in (
            "body_only",
            "uniform_ray",
            "conditional_resolve",
        ):
            raise ValueError("unsupported stage_one_trust_mode")
        if self.damping_mode not in ("isotropic", "response_scaled"):
            raise ValueError("unsupported damping_mode")

    @property
    def starts(self):
        return list(range(0, self.frames - self.horizon + 1, self.stride))

    @property
    def observation_channels(self):
        return {"cahn_hilliard": 2, "gray_scott": 4, "kolmogorov_ns": 6, "mhd2d": 8}[self.case_name]

    @property
    def action_channels(self):
        return {"cahn_hilliard": 1, "gray_scott": 1, "kolmogorov_ns": 2, "mhd2d": 3}[self.case_name]

    @property
    def field_channels(self):
        return {"cahn_hilliard": 1, "gray_scott": 2, "kolmogorov_ns": 1, "mhd2d": 2}[self.case_name]

    @property
    def body_geometry_policy(self):
        return "own_g_output_orthonormal"

    @property
    def executed_map_policy(self):
        return (
            f"nonlinear_anchor_refit_rank_{self.body_compensation_rank}"
            if self.body_compensation_rank > 0
            else "additive"
        )

    @property
    def head_penalty_policy(self):
        return (
            "readout_l2_with_forward_acceptance" if self.head_weight_penalty_eta > 0.0 else "none"
        )

    def dictionary(self):
        return {
            **asdict(self),
            "starts": self.starts,
            "time_horizon": self.horizon * self.sample_dt,
            "body_geometry_policy": self.body_geometry_policy,
            "executed_map_policy": self.executed_map_policy,
            "head_penalty_policy": self.head_penalty_policy,
        }


CH_RECIPE = Recipe(
    case_name="cahn_hilliard",
    protocol=CANONICAL_CASE_PROTOCOLS["cahn_hilliard"],
    ppo_lr=1e-4,
    ad_lr=3e-5,
    ad_epochs=128,
)
GS_RECIPE = replace(
    CH_RECIPE,
    protocol=CANONICAL_CASE_PROTOCOLS["gray_scott"],
    case_name="gray_scott",
    frames=1000,
    sample_dt=10.0,
    ppo_lr=3e-5,
    ad_lr=3e-5,
    ad_epochs=128,
    closure_rl_lr=1e-4,
    window_budget_anchor=736756,
    body_basis_mode="dense",
    body_compensation_rank=4,
    stage_one_trust_mode="conditional_resolve",
    damping_mode="response_scaled",
    head_weight_penalty_eta=1e-8,
)
NS_RECIPE = replace(
    CH_RECIPE,
    protocol=CANONICAL_CASE_PROTOCOLS["kolmogorov_ns"],
    case_name="kolmogorov_ns",
    frames=160,
    sample_dt=0.02,
    fine_rounds=100,
    patience=5,
    training_groups_per_round=16,
    ppo_lr=1e-5,
    ad_lr=3e-3,
    ad_epochs=128,
    closure_rl_lr=3e-5,
    window_budget_anchor=283228,
    head_weight_penalty_eta=1e-8,
)
MHD_RECIPE = replace(
    CH_RECIPE,
    protocol=CANONICAL_CASE_PROTOCOLS["mhd2d"],
    case_name="mhd2d",
    frames=160,
    sample_dt=0.025,
    fine_rounds=100,
    patience=5,
    training_groups_per_round=16,
    ppo_lr=1e-5,
    ad_lr=3e-3,
    ad_epochs=128,
    closure_rl_lr=3e-5,
    window_budget_anchor=278544,
    head_weight_penalty_eta=1e-8,
)
RECIPE = CH_RECIPE
RECIPE_BY_CASE = {r.case_name: r for r in (CH_RECIPE, GS_RECIPE, NS_RECIPE, MHD_RECIPE)}
PROTOCOLS = {r.protocol: r for r in RECIPE_BY_CASE.values()}


def trial_map(parent_count, recipe=RECIPE):
    if parent_count % recipe.group_size:
        raise ValueError("canonical four-parent groups required")
    return [(t, g) for t in recipe.starts for g in range(parent_count // recipe.group_size)]


def epoch_batches(trials, epoch, seed=0, batch_groups=2):
    order = np.random.default_rng(np.random.SeedSequence([seed, 110, epoch])).permutation(trials)
    return [order[k : k + batch_groups].tolist() for k in range(0, len(order), batch_groups)]


def validate_profile_data(data: Any, *, case_name: str | None = None) -> tuple[str, dict[str, Any]]:
    """Validate a parsed 2D JSON benchmark profile and resolve its target case."""
    if not isinstance(data, dict):
        raise ValueError("2D config profile must be a JSON object")
    unknown_top = sorted(set(data.keys()) - ALLOWED_PROFILE_TOP_KEYS)
    if unknown_top:
        raise ValueError(f"Unknown top-level section(s) in 2D config profile: {unknown_top}")

    top_case = data.get("case_name")
    alt_case = data.get("case")
    if top_case is not None and alt_case is not None and top_case != alt_case:
        raise ValueError(
            f"Conflicting top-level case identifiers: case_name={top_case!r} != case={alt_case!r}"
        )
    profile_case = top_case if top_case is not None else alt_case

    meta = data.get("benchmark_metadata")
    if meta is not None:
        if not isinstance(meta, dict):
            raise ValueError("benchmark_metadata must be a JSON object")
        unknown_meta = sorted(set(meta.keys()) - ALLOWED_BENCHMARK_METADATA_KEYS)
        if unknown_meta:
            raise ValueError(f"Unknown field(s) in benchmark_metadata: {unknown_meta}")
        meta_case = meta.get("case_name")
        if profile_case is not None and meta_case is not None and meta_case != profile_case:
            raise ValueError(
                f"Conflict between profile case_name={profile_case!r} and benchmark_metadata.case_name={meta_case!r}"
            )
        if profile_case is None and meta_case is not None:
            profile_case = meta_case

        wf = meta.get("default_initialization_workflow")
        if wf is not None and wf not in SUPPORTED_WORKFLOWS:
            raise ValueError(
                f"Unsupported default_initialization_workflow {wf!r}; expected one of {sorted(SUPPORTED_WORKFLOWS)}"
            )
        sup_wf = meta.get("supported_workflows")
        if sup_wf is not None:
            if (
                not isinstance(sup_wf, list)
                or not sup_wf
                or any(w not in SUPPORTED_WORKFLOWS for w in sup_wf)
            ):
                raise ValueError(
                    f"supported_workflows must be a non-empty list of {sorted(SUPPORTED_WORKFLOWS)}"
                )
            if wf is not None and wf not in sup_wf:
                raise ValueError(
                    f"default_initialization_workflow {wf!r} must be listed in supported_workflows {sup_wf!r}"
                )

    if case_name is not None and profile_case is not None and case_name != profile_case:
        raise ValueError(
            f"Conflict between requested case {case_name!r} and profile case {profile_case!r}"
        )
    resolved_case = case_name or profile_case
    if resolved_case not in RECIPE_BY_CASE:
        raise ValueError(f"unknown 2D benchmark case in profile: {resolved_case!r}")

    raw_overrides = data.get("recipe_overrides", {})
    if not isinstance(raw_overrides, dict):
        raise ValueError("recipe_overrides must be a JSON object")
    valid_keys = {f.name for f in fields(Recipe)}
    unknown_overrides = sorted(set(raw_overrides.keys()) - valid_keys)
    if unknown_overrides:
        raise ValueError(f"Unknown field(s) in recipe_overrides: {unknown_overrides}")
    if "case_name" in raw_overrides and raw_overrides["case_name"] != resolved_case:
        raise ValueError(
            f"Conflict between resolved case {resolved_case!r} and recipe_overrides.case_name={raw_overrides['case_name']!r}"
        )

    cli_defaults = data.get("cli_defaults")
    if cli_defaults is not None:
        if not isinstance(cli_defaults, dict) or any(
            not isinstance(k, str) or not k.startswith("--") for k in cli_defaults
        ):
            raise ValueError("cli_defaults must be a JSON object keyed by '--flag' strings")
        unknown_cli = sorted(set(cli_defaults.keys()) - ALLOWED_CLI_DEFAULTS_KEYS)
        if unknown_cli:
            raise ValueError(f"Unknown flag(s) in cli_defaults: {unknown_cli}")

    return str(resolved_case), data


def apply_validated_recipe_overrides(base: Recipe, overrides: dict[str, Any]) -> Recipe:
    if not isinstance(overrides, dict):
        raise ValueError("recipe_overrides must be a dict")
    valid_keys = {f.name for f in fields(base)}
    unknown = sorted(set(overrides.keys()) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown Recipe override field(s): {unknown}")
    return replace(base, **overrides) if overrides else base


def _verify_metadata_matches_recipe(meta: dict[str, Any], resolved: Recipe) -> None:
    if "case_name" in meta and (
        not isinstance(meta["case_name"], str) or meta["case_name"] != resolved.case_name
    ):
        raise ValueError(
            f"Conflict between benchmark_metadata['case_name']={meta['case_name']!r} and resolved recipe value {resolved.case_name!r}"
        )
    int_checks = (
        ("window_horizon", resolved.horizon),
        ("fine_rounds", resolved.fine_rounds),
        ("directions", resolved.directions),
        ("training_groups_per_round", resolved.training_groups_per_round),
        ("window_budget_anchor", resolved.window_budget_anchor),
    )
    for meta_key, rec_val in int_checks:
        if meta_key in meta:
            if not _is_strict_int(meta[meta_key]) or int(meta[meta_key]) != rec_val:
                raise ValueError(
                    f"Conflict between benchmark_metadata[{meta_key!r}]={meta[meta_key]!r} and resolved recipe value {rec_val!r}"
                )
    if "sample_dt" in meta:
        m_dt = meta["sample_dt"]
        if not _is_strict_real(m_dt) or not math.isclose(
            float(m_dt), float(resolved.sample_dt), rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError(
                f"Conflict between benchmark_metadata['sample_dt']={m_dt!r} and resolved recipe.sample_dt={resolved.sample_dt!r}"
            )
    if "closure_parameters" in meta:
        expected_params = _EXPECTED_CASE_CLOSURE_PARAMETERS[resolved.case_name]
        if (
            not _is_strict_int(meta["closure_parameters"])
            or int(meta["closure_parameters"]) != expected_params
        ):
            raise ValueError(
                f"Conflict between benchmark_metadata['closure_parameters']={meta['closure_parameters']!r} and canonical parameter count {expected_params}"
            )


def load_recipe_profile(config_path: str | Path, case_name: str | None = None) -> Recipe:
    raw_data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    resolved_case, data = validate_profile_data(raw_data, case_name=case_name)
    base = RECIPE_BY_CASE[resolved_case]
    resolved = apply_validated_recipe_overrides(base, data.get("recipe_overrides", {}))
    meta = data.get("benchmark_metadata")
    if isinstance(meta, dict):
        _verify_metadata_matches_recipe(meta, resolved)
    return resolved


def configure(recipe: Recipe) -> dict[str, Any]:
    """Configure G's numerical execution flags, independent of physical F/H internals."""
    if recipe.strict_actor_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    return {
        "mode": "strict_FP32_actor" if recipe.strict_actor_fp32 else "legacy_default_flags",
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_version": str(torch.__version__),
        "scope": "learner-owned neural computation only; no physical operator/configuration exposed",
    }
