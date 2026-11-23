"""Configuration contract and JSON profile validation for 3D Double-Gyre POM."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from .ocean.contracts import resolve_env_var

DEFAULT_DATASET_ROOT_STR = "./data/pom3d_double_gyre"
DEFAULT_OUTPUT_ROOT_STR = "./runs/pom3d"
DEFAULT_DATASET_ROOT = Path(DEFAULT_DATASET_ROOT_STR)
DEFAULT_OUTPUT_ROOT = Path(DEFAULT_OUTPUT_ROOT_STR)

VALID_PROTOCOLS: tuple[str, ...] = ("pure_4h",)
VALID_ABLATION_VARIANTS: tuple[str, ...] = (
    "none",
    "no_stage1",
    "no_body_comp",
    "no_kobs",
    "intra_hour_only",
    "action_l2",
    "head_only_stage2",
)

ALLOWED_PROFILE_TOP_KEYS: frozenset[str] = frozenset(
    {"protocol", "benchmark_metadata", "config_overrides"}
)

ALLOWED_BENCHMARK_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "case_name",
        "initialization",
        "grid",
        "active_vertical_levels",
        "in_channels",
        "out_channels",
        "closure_parameters",
        "readout_parameters",
        "dti_seconds",
        "dte_seconds",
        "isplit",
        "ra_filter_smoth",
        "hours_per_window",
        "segments_per_window",
    }
)

NON_OVERRIDABLE_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "source_version",
        "source_sha256",
        "source_tree_sha256",
        "effective_config_sha256",
        "source_bundle_sha256",
        "common_hash",
        "forge_stage2_steps",
    }
)


def _require_str(name: str, value: Any, *, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {value!r}")
    if non_empty and not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(
    name: str,
    value: Any,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    ival = int(value)
    if min_val is not None and ival < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {ival}")
    if max_val is not None and ival > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {ival}")
    return ival


def _require_finite_float(
    name: str,
    value: Any,
    *,
    strictly_positive: bool = False,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if strictly_positive and fval <= 0.0:
        raise ValueError(f"{name} must be > 0, got {fval}")
    if min_val is not None and fval < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {fval}")
    if max_val is not None and fval > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {fval}")
    return fval


def _require_int_tuple(
    name: str,
    value: Any,
    *,
    non_empty: bool = True,
    unique: bool = False,
    min_val: int | None = None,
    max_val: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of integers, got {value!r}")
    items = tuple(
        _require_int(f"{name}[{idx}]", item, min_val=min_val, max_val=max_val)
        for idx, item in enumerate(value)
    )
    if non_empty and not items:
        raise ValueError(f"{name} must be non-empty")
    if unique and len(set(items)) != len(items):
        raise ValueError(f"{name} must not contain duplicate entries, got {items!r}")
    return items


def _require_float_tuple(
    name: str,
    value: Any,
    *,
    non_empty: bool = True,
    strictly_positive: bool = True,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of numbers, got {value!r}")
    items = tuple(
        _require_finite_float(f"{name}[{idx}]", item, strictly_positive=strictly_positive)
        for idx, item in enumerate(value)
    )
    if non_empty and not items:
        raise ValueError(f"{name} must be non-empty")
    return items


def resolve_default_dataset_root(environ: Mapping[str, str] | None = None) -> Path:
    val = resolve_env_var(
        "FORGE_POM3D_DATASET_ROOT", DEFAULT_DATASET_ROOT_STR, environ=environ
    ).strip()
    return Path(val or DEFAULT_DATASET_ROOT_STR)


def resolve_default_output_root(environ: Mapping[str, str] | None = None) -> Path:
    val = resolve_env_var(
        "FORGE_POM3D_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT_STR, environ=environ
    ).strip()
    return Path(val or DEFAULT_OUTPUT_ROOT_STR)


def _count_expected_pom_net_params(
    in_channels: int, hidden_channels: int, out_channels: int, num_blocks: int
) -> tuple[int, int]:
    """Compute exact `(total_params, readout_params)` for `ForgePOMNet` architecture."""
    stem = in_channels * hidden_channels + hidden_channels + 2 * hidden_channels
    c = hidden_channels
    c_q = c // 4
    per_block = (
        2 * c
        + 2 * (c * (c // 4) * 9 + c)
        + 2 * c
        + (2 * c * c + c)
        + (c * c + c)
        + (c * c_q + c_q)
        + (c_q * c + c)
        + 1
    )
    trunk_norm = 2 * hidden_channels
    head = out_channels * hidden_channels + out_channels
    total = stem + num_blocks * per_block + trunk_norm + head
    return total, head


@dataclass
class ForgePOM3DConfig:
    """Configuration for 3D Double-Gyre POM FORGE experiments under the Pure-4h Protocol."""

    dataset_root: str = DEFAULT_DATASET_ROOT_STR
    output_root: str = DEFAULT_OUTPUT_ROOT_STR
    input_stats_path: str = ""
    seed: int = 0
    protocol_mode: str = "pure_4h"

    # Grid & solver constants (matching semi_implicit double_gyre coarse configuration)
    im: int = 90
    jm: int = 82
    kb: int = 16
    nz_active: int = 15
    dte: float = 20.0
    dti: float = 600.0
    isplit: int = 30
    substeps_per_hour: int = 6
    hours_per_window: int = 4
    segments_per_window: int = 12
    substep_divisor: float = 6.0

    # Provenance & non-self-referential source/config digests (populated at run time)
    source_version: str = ""
    source_sha256: str = ""
    source_tree_sha256: str = ""
    effective_config_sha256: str = ""
    source_bundle_sha256: str = ""
    common_hash: str = ""

    # Query budget & pure-4h validation selection
    max_window_budget: float | None = 121504.0
    planned_window_budget: float = 121504.0
    val_improve_tol: float = 1e-6

    # Response operator held-out gate & gain bounds
    response_gate_rel_err_max: float = 0.50
    response_gate_cosine_min: float = 0.80
    response_gate_r2_min: float = 0.50
    response_gate_per_probe_max: float = 1.00
    forge_full_gain_bound: float = 4.0
    obs_response_gain_bound: float = 6.0
    obs_response_cosine_min: float = 0.50
    obs_response_r2_min: float = 0.15
    ppo_target_kl: float = 0.05

    # Climatological standard deviations (used for instantaneous NRMSE reporting)
    sigma_u: float = 0.0872255265712738
    sigma_v: float = 0.06801831722259521
    sigma_t: float = 6.5356221199035645
    sigma_s: float = 0.2586655914783478
    sigma_el: float = 0.1456301063299179

    # Calibrated 1-hour physical residual RMS scales
    scale_u: float = 0.0015076317471964318
    scale_v: float = 0.0016684176785869917
    scale_t: float = 0.0027521051359013214
    scale_s: float = 0.00016846091935662978
    action_gain: float = 1.0
    action_reg_weight: float = 0.02

    # Balanced 4-hour trajectory loss normalization scales
    loss_scale_u: float = 0.00242213
    loss_scale_v: float = 0.00260602
    loss_scale_t: float = 0.00881092
    loss_scale_s: float = 0.00080976
    loss_scale_el: float = 0.01034688

    # Data split & sampling protocol (member-isolated: train=0..5, val=6, test_id=7)
    spinup_days: int = 1000
    train_members: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    train_start_day: int = 1000
    train_end_day: int = 7000
    train_stride_days: int = 20

    # Validation split (16 independent 4h windows; sole selection metric)
    val_members: tuple[int, ...] = (6,)
    val_start_day: int = 7000
    val_end_day: int = 7960
    val_stride_days: int = 60

    # Frozen held-out Test ID split on member 7 (48 strided windows + 8 continuity starts)
    eval_member: int = 7
    test_start_day: int = 8000
    test_end_day: int = 8960
    test_stride_days: int = 20
    eval_starts: tuple[int, ...] = (8000, 8064, 8128, 8192, 8256, 8320, 8384, 8448)

    # Network architecture (161 -> 256 -> 60 with 257-dim linear readout head)
    in_channels: int = 161
    hidden_channels: int = 256
    out_channels: int = 60
    num_blocks: int = 8

    # Stage 2: FORGE Main Two-Stage & Stage-I-Only Black-Box Post-Training (from common_zero.pt)
    ablation_variant: str = "none"
    forge_rounds: int = 800
    forge_patience: int = 20
    forge_windows_per_gpu: int = 1
    forge_screen_windows_per_gpu: int = 2
    forge_num_directions: int = 8
    forge_curv_rank: int = 4
    forge_comp_rank: int = 4
    forge_probe_eps: float = 0.35
    forge_stage2_probe_amp: float = 0.20
    forge_ridge_damping: float = 1e-2
    forge_head_step_grid: tuple[float, ...] = (0.20, 0.50, 1.00)
    forge_response_ridge: float = 5e-2
    forge_action_reg_weight: float = 0.015
    forge_cg_trust_rms: float = 0.25
    forge_cg_trust_min: float = 0.08
    forge_cg_trust_max: float = 0.65
    forge_stage2_replay_rounds: int = 8
    forge_stage2_replay_decay: float = 0.82
    forge_stage2_minibatch_windows: int = 4
    forge_proj_steps: int = 60
    forge_stage2_steps: int | None = field(default=None, repr=False)
    forge_stage2_head_lr: float = 1.5e-4
    forge_stage2_body_lr: float = 4.0e-5
    forge_anchor_weight: float = 0.02

    # Stage 3..5: Black-Box Baselines (ES, EKI, PPO) from common_zero.pt
    baseline_rounds: int = 1600
    baseline_patience: int = 20
    anchor_window_budget: float = 121504.0
    es_eki_num_directions: int = 16
    es_sigma: float = 0.08
    es_step_radii: tuple[float, ...] = (0.60, 0.30, 0.15, 0.05)
    eki_ensemble_size: int = 24
    eki_sigma: float = 0.08
    eki_reg: float = 0.05
    eki_step_radii: tuple[float, ...] = (1.00, 0.50, 0.25, 0.10)
    ppo_clip_eps: float = 0.20
    ppo_lr: float = 5e-7
    ppo_explore_std: float = 0.05
    ppo_epochs: int = 4

    # Stage 6: Differentiable Neural-Emulator-Assisted Policy Training
    emulator_windows_per_gpu: int = 1
    emulator_probe_amp: float = 0.16
    emulator_width: int = 96
    emulator_blocks: int = 4
    emulator_lr: float = 3e-4
    emulator_warmup_updates: int = 80
    emulator_updates: int = 40
    emulator_buffer_rounds: int = 2
    emulator_policy_updates: int = 12
    emulator_policy_head_lr: float = 1e-4
    emulator_policy_body_lr: float = 3e-5
    emulator_step_alphas: tuple[float, ...] = (1.0, 0.5, 0.25)
    emulator_param_trust_weight: float = 1e-3
    emulator_action_trust_sigma: float = 0.08

    # Stage 7: Public-Interface Closure-RL
    closure_rl_windows_per_gpu: int = 2
    closure_rl_explore_std: float = 0.05
    closure_rl_clip_eps: float = 0.20
    closure_rl_epochs: int = 4
    closure_rl_minibatch_frames: int = 4
    closure_rl_lr: float = 2e-5
    closure_rl_critic_lr: float = 3e-4
    closure_rl_critic_width: int = 64
    closure_rl_target_kl: float = 0.02
    closure_rl_max_example_kl: float = 0.25
    closure_rl_step_alphas: tuple[float, ...] = (1.0, 0.5)

    # Stage 8: Privileged White-Box Full-24-Substep Solver BPTT Reference (up to 300 epochs)
    ad_max_epochs: int = 300
    ad_steps_per_epoch: int = 45
    ad_head_lr: float = 1.5e-4
    ad_body_lr: float = 4.0e-5
    ad_weight_decay: float = 1e-4
    ad_grad_clip: float = 1.0
    ad_patience: int = 15
    ad_lr_patience: int = 5
    ad_lr_factor: float = 0.5
    ad_min_lr: float = 1e-6

    def __post_init__(self) -> None:
        self._normalize_and_validate_self()

    def _normalize_and_validate_self(self) -> None:
        self.dataset_root = _require_str("dataset_root", self.dataset_root, non_empty=True)
        self.output_root = _require_str("output_root", self.output_root, non_empty=True)
        self.input_stats_path = _require_str("input_stats_path", self.input_stats_path)
        self.protocol_mode = _require_str("protocol_mode", self.protocol_mode, non_empty=True)
        self.ablation_variant = _require_str(
            "ablation_variant", self.ablation_variant, non_empty=True
        )
        for str_field in (
            "source_version",
            "source_sha256",
            "source_tree_sha256",
            "effective_config_sha256",
            "source_bundle_sha256",
            "common_hash",
        ):
            setattr(self, str_field, _require_str(str_field, getattr(self, str_field)))

        if self.protocol_mode not in VALID_PROTOCOLS:
            raise ValueError(f"Unsupported protocol_mode={self.protocol_mode!r}")
        if self.ablation_variant not in VALID_ABLATION_VARIANTS:
            raise ValueError(
                f"Invalid ablation_variant={self.ablation_variant!r}; "
                f"expected one of {VALID_ABLATION_VARIANTS}"
            )

        self.seed = _require_int("seed", self.seed, min_val=0)

        # Grid & vertical discretization
        self.im = _require_int("im", self.im, min_val=5)
        self.jm = _require_int("jm", self.jm, min_val=5)
        self.kb = _require_int("kb", self.kb)
        self.nz_active = _require_int("nz_active", self.nz_active)
        if self.kb != 16 or self.nz_active != 15:
            raise ValueError(
                f"Invalid 3D POM vertical configuration: kb={self.kb}, nz_active={self.nz_active} (expected 16, 15)"
            )

        # Semi-implicit time stepping & window temporal consistency
        self.dte = _require_finite_float("dte", self.dte, strictly_positive=True)
        self.dti = _require_finite_float("dti", self.dti, strictly_positive=True)
        self.isplit = _require_int("isplit", self.isplit, min_val=1)
        if (
            abs(self.dti - 600.0) > 1e-9
            or abs(self.dte - 20.0) > 1e-9
            or self.isplit != 30
            or abs(self.dti - self.dte * float(self.isplit)) > 1e-9
        ):
            raise ValueError(
                f"Invalid semi-implicit time split: dti={self.dti}, dte={self.dte}, isplit={self.isplit}"
            )

        self.substeps_per_hour = _require_int(
            "substeps_per_hour", self.substeps_per_hour, min_val=1
        )
        self.hours_per_window = _require_int(
            "hours_per_window", self.hours_per_window, min_val=1, max_val=4
        )
        self.segments_per_window = _require_int(
            "segments_per_window", self.segments_per_window, min_val=1
        )
        self.substep_divisor = _require_finite_float(
            "substep_divisor", self.substep_divisor, strictly_positive=True
        )
        if abs(self.substeps_per_hour * self.dti - 3600.0) > 1e-9:
            raise ValueError(
                f"substeps_per_hour={self.substeps_per_hour} inconsistent with dti={self.dti}s"
            )
        if abs(self.substep_divisor - float(self.substeps_per_hour)) > 1e-9:
            raise ValueError(
                f"substep_divisor={self.substep_divisor} must equal substeps_per_hour={self.substeps_per_hour}"
            )
        if self.segments_per_window != 3 * self.hours_per_window:
            raise ValueError(
                f"segments_per_window={self.segments_per_window} must equal "
                f"3 * hours_per_window ({3 * self.hours_per_window}) for 20-minute reference segments"
            )
        if self.protocol_mode == "pure_4h" and (
            self.hours_per_window != 4 or self.segments_per_window != 12
        ):
            raise ValueError(
                f"protocol_mode='pure_4h' requires hours_per_window=4 and segments_per_window=12, "
                f"got ({self.hours_per_window}, {self.segments_per_window})"
            )

        # Stage II step alias normalization (strict int check BEFORE int conversion)
        proj_steps = _require_int("forge_proj_steps", self.forge_proj_steps, min_val=1)
        if self.forge_stage2_steps is not None:
            stage2_steps = _require_int("forge_stage2_steps", self.forge_stage2_steps, min_val=1)
            if proj_steps != 60 and proj_steps != stage2_steps:
                raise ValueError(
                    f"Conflicting forge_proj_steps={proj_steps} and forge_stage2_steps={stage2_steps}"
                )
            proj_steps = stage2_steps
        self.forge_proj_steps = proj_steps
        self.forge_stage2_steps = proj_steps

        # Budgets
        if self.max_window_budget is not None:
            self.max_window_budget = _require_finite_float(
                "max_window_budget", self.max_window_budget, strictly_positive=True
            )
            self.planned_window_budget = float(self.max_window_budget)
        else:
            self.planned_window_budget = _require_finite_float(
                "planned_window_budget", self.planned_window_budget, strictly_positive=True
            )

        # Member splits (strict type check BEFORE tuple conversion + pairwise disjointness)
        self.train_members = _require_int_tuple(
            "train_members", self.train_members, non_empty=True, unique=True, min_val=0, max_val=7
        )
        self.val_members = _require_int_tuple(
            "val_members", self.val_members, non_empty=True, unique=True, min_val=0, max_val=7
        )
        self.eval_member = _require_int("eval_member", self.eval_member, min_val=0, max_val=7)

        overlap_train_val = sorted(set(self.train_members) & set(self.val_members))
        if overlap_train_val:
            raise ValueError(
                f"Data split leakage: train_members {self.train_members} and "
                f"val_members {self.val_members} overlap on {overlap_train_val}"
            )
        if self.eval_member in self.train_members:
            raise ValueError(
                f"Data split leakage: eval_member={self.eval_member} overlaps with train_members={self.train_members}"
            )
        if self.eval_member in self.val_members:
            raise ValueError(
                f"Data split leakage: eval_member={self.eval_member} overlaps with val_members={self.val_members}"
            )

        # Model day ranges & sampling strides (4h windows require intraday labels in [1, 8999])
        self.spinup_days = _require_int("spinup_days", self.spinup_days, min_val=0, max_val=8999)
        min_allowed_day = max(1, self.spinup_days)
        for prefix in ("train", "val", "test"):
            s_day = _require_int(
                f"{prefix}_start_day", getattr(self, f"{prefix}_start_day"), min_val=1, max_val=8999
            )
            e_day = _require_int(
                f"{prefix}_end_day", getattr(self, f"{prefix}_end_day"), min_val=2, max_val=9000
            )
            stride = _require_int(
                f"{prefix}_stride_days", getattr(self, f"{prefix}_stride_days"), min_val=1
            )
            if s_day < min_allowed_day:
                raise ValueError(
                    f"{prefix}_start_day={s_day} precedes spinup_days={self.spinup_days}"
                )
            if s_day >= e_day:
                raise ValueError(f"{prefix}_start_day={s_day} must be < {prefix}_end_day={e_day}")
            last_gen_day = s_day + ((e_day - 1 - s_day) // stride) * stride
            if last_gen_day > 8999:
                raise ValueError(
                    f"{prefix} window schedule generates start_day={last_gen_day} > 8999 "
                    f"(intraday 4h reference labels only cover days 0..8999)"
                )
            setattr(self, f"{prefix}_start_day", s_day)
            setattr(self, f"{prefix}_end_day", e_day)
            setattr(self, f"{prefix}_stride_days", stride)

        self.eval_starts = _require_int_tuple(
            "eval_starts",
            self.eval_starts,
            non_empty=True,
            unique=True,
            min_val=min_allowed_day,
            max_val=8999,
        )
        if any(d < self.test_start_day or d > 8999 for d in self.eval_starts):
            raise ValueError(
                f"eval_starts={self.eval_starts} must lie within [test_start_day={self.test_start_day}, 8999]"
            )

        # Network architecture
        self.in_channels = _require_int("in_channels", self.in_channels, min_val=1)
        self.hidden_channels = _require_int("hidden_channels", self.hidden_channels, min_val=4)
        self.out_channels = _require_int("out_channels", self.out_channels, min_val=1)
        self.num_blocks = _require_int("num_blocks", self.num_blocks, min_val=1)
        if self.hidden_channels % 4 != 0:
            raise ValueError(
                f"hidden_channels={self.hidden_channels} must be divisible by 4 for spectral block bottlenecks"
            )
        if self.in_channels != 161 or self.out_channels != 4 * self.nz_active:
            raise ValueError(
                f"Invalid 3D POM channel interface: in_channels={self.in_channels} (expected 161), "
                f"out_channels={self.out_channels} (expected {4 * self.nz_active})"
            )

        # Positive integer algorithm parameters
        pos_int_fields = (
            "forge_rounds",
            "forge_patience",
            "forge_windows_per_gpu",
            "forge_screen_windows_per_gpu",
            "forge_num_directions",
            "forge_curv_rank",
            "forge_stage2_replay_rounds",
            "forge_stage2_minibatch_windows",
            "baseline_rounds",
            "baseline_patience",
            "es_eki_num_directions",
            "ppo_epochs",
            "emulator_windows_per_gpu",
            "emulator_width",
            "emulator_blocks",
            "emulator_warmup_updates",
            "emulator_updates",
            "emulator_buffer_rounds",
            "emulator_policy_updates",
            "closure_rl_windows_per_gpu",
            "closure_rl_epochs",
            "closure_rl_minibatch_frames",
            "closure_rl_critic_width",
            "ad_max_epochs",
            "ad_steps_per_epoch",
            "ad_patience",
            "ad_lr_patience",
        )
        for fname in pos_int_fields:
            setattr(self, fname, _require_int(fname, getattr(self, fname), min_val=1))

        self.forge_comp_rank = _require_int("forge_comp_rank", self.forge_comp_rank, min_val=0)
        self.eki_ensemble_size = _require_int(
            "eki_ensemble_size", self.eki_ensemble_size, min_val=2
        )
        if self.forge_curv_rank > self.forge_num_directions:
            raise ValueError(
                f"forge_curv_rank={self.forge_curv_rank} cannot exceed forge_num_directions={self.forge_num_directions}"
            )

        # Strictly positive finite float parameters
        pos_float_fields = (
            "response_gate_rel_err_max",
            "response_gate_per_probe_max",
            "forge_full_gain_bound",
            "obs_response_gain_bound",
            "ppo_target_kl",
            "sigma_u",
            "sigma_v",
            "sigma_t",
            "sigma_s",
            "sigma_el",
            "scale_u",
            "scale_v",
            "scale_t",
            "scale_s",
            "action_gain",
            "loss_scale_u",
            "loss_scale_v",
            "loss_scale_t",
            "loss_scale_s",
            "loss_scale_el",
            "forge_probe_eps",
            "forge_stage2_probe_amp",
            "forge_ridge_damping",
            "forge_response_ridge",
            "forge_cg_trust_rms",
            "forge_cg_trust_min",
            "forge_cg_trust_max",
            "forge_stage2_replay_decay",
            "forge_stage2_head_lr",
            "forge_stage2_body_lr",
            "anchor_window_budget",
            "es_sigma",
            "eki_sigma",
            "eki_reg",
            "ppo_clip_eps",
            "ppo_lr",
            "ppo_explore_std",
            "emulator_probe_amp",
            "emulator_lr",
            "emulator_policy_head_lr",
            "emulator_policy_body_lr",
            "emulator_action_trust_sigma",
            "closure_rl_explore_std",
            "closure_rl_clip_eps",
            "closure_rl_lr",
            "closure_rl_critic_lr",
            "closure_rl_target_kl",
            "closure_rl_max_example_kl",
            "ad_head_lr",
            "ad_body_lr",
            "ad_grad_clip",
            "ad_lr_factor",
            "ad_min_lr",
        )
        for fname in pos_float_fields:
            setattr(
                self,
                fname,
                _require_finite_float(fname, getattr(self, fname), strictly_positive=True),
            )

        # Non-negative / bounded finite float parameters
        nonneg_float_fields = (
            "val_improve_tol",
            "action_reg_weight",
            "forge_action_reg_weight",
            "forge_anchor_weight",
            "emulator_param_trust_weight",
            "ad_weight_decay",
        )
        for fname in nonneg_float_fields:
            setattr(self, fname, _require_finite_float(fname, getattr(self, fname), min_val=0.0))

        self.response_gate_cosine_min = _require_finite_float(
            "response_gate_cosine_min", self.response_gate_cosine_min, min_val=-1.0, max_val=1.0
        )
        self.response_gate_r2_min = _require_finite_float(
            "response_gate_r2_min", self.response_gate_r2_min, max_val=1.0
        )
        self.obs_response_cosine_min = _require_finite_float(
            "obs_response_cosine_min", self.obs_response_cosine_min, min_val=-1.0, max_val=1.0
        )
        self.obs_response_r2_min = _require_finite_float(
            "obs_response_r2_min", self.obs_response_r2_min, max_val=1.0
        )

        if not (self.forge_cg_trust_min <= self.forge_cg_trust_rms <= self.forge_cg_trust_max):
            raise ValueError(
                f"Invalid CG trust-region bounds: min={self.forge_cg_trust_min}, "
                f"rms={self.forge_cg_trust_rms}, max={self.forge_cg_trust_max}"
            )
        if self.forge_stage2_replay_decay > 1.0:
            raise ValueError(
                f"forge_stage2_replay_decay={self.forge_stage2_replay_decay} must be <= 1.0"
            )
        if self.ad_lr_factor >= 1.0:
            raise ValueError(f"ad_lr_factor={self.ad_lr_factor} must lie in (0, 1)")

        # Step-grid / radii / alpha sequences
        self.forge_head_step_grid = _require_float_tuple(
            "forge_head_step_grid", self.forge_head_step_grid
        )
        self.es_step_radii = _require_float_tuple("es_step_radii", self.es_step_radii)
        self.eki_step_radii = _require_float_tuple("eki_step_radii", self.eki_step_radii)
        self.emulator_step_alphas = _require_float_tuple(
            "emulator_step_alphas", self.emulator_step_alphas
        )
        self.closure_rl_step_alphas = _require_float_tuple(
            "closure_rl_step_alphas", self.closure_rl_step_alphas
        )

    def validate(self, benchmark_metadata: Mapping[str, Any] | None = None) -> None:
        self._normalize_and_validate_self()
        if benchmark_metadata is not None:
            if not isinstance(benchmark_metadata, Mapping):
                raise ValueError("benchmark_metadata must be a mapping")
            if benchmark_metadata:
                self._validate_against_metadata(benchmark_metadata)

    def _validate_against_metadata(self, meta: Mapping[str, Any]) -> None:
        unknown_meta = set(meta) - ALLOWED_BENCHMARK_METADATA_KEYS
        if unknown_meta:
            raise ValueError(f"Unknown keys in benchmark_metadata: {sorted(unknown_meta)}")
        if "case_name" in meta and meta["case_name"] != "pom3d_double_gyre":
            raise ValueError(f"Unsupported benchmark_metadata.case_name={meta['case_name']!r}")
        if "initialization" in meta and meta["initialization"] != "zero_residual_head":
            raise ValueError(
                f"Unsupported benchmark_metadata.initialization={meta['initialization']!r}"
            )
        if "grid" in meta:
            grid_seq = _require_int_tuple("benchmark_metadata.grid", meta["grid"], min_val=1)
            if list(grid_seq) != [self.im, self.jm, self.kb]:
                raise ValueError(
                    f"benchmark_metadata.grid={meta['grid']!r} conflicts with "
                    f"effective grid {[self.im, self.jm, self.kb]}"
                )
        int_meta_checks = (
            ("active_vertical_levels", self.nz_active),
            ("in_channels", self.in_channels),
            ("out_channels", self.out_channels),
            ("isplit", self.isplit),
            ("hours_per_window", self.hours_per_window),
            ("segments_per_window", self.segments_per_window),
        )
        for key, eff_val in int_meta_checks:
            if key in meta:
                m_val = _require_int(f"benchmark_metadata.{key}", meta[key], min_val=1)
                if m_val != eff_val:
                    raise ValueError(
                        f"benchmark_metadata.{key}={meta[key]!r} conflicts with effective {eff_val}"
                    )
        if "dti_seconds" in meta:
            m_dti = _require_finite_float(
                "benchmark_metadata.dti_seconds", meta["dti_seconds"], strictly_positive=True
            )
            if abs(m_dti - self.dti) > 1e-12:
                raise ValueError(
                    f"benchmark_metadata.dti_seconds={meta['dti_seconds']!r} conflicts with dti={self.dti}"
                )
        if "dte_seconds" in meta:
            m_dte = _require_finite_float(
                "benchmark_metadata.dte_seconds", meta["dte_seconds"], strictly_positive=True
            )
            if abs(m_dte - self.dte) > 1e-12:
                raise ValueError(
                    f"benchmark_metadata.dte_seconds={meta['dte_seconds']!r} conflicts with dte={self.dte}"
                )
        if "ra_filter_smoth" in meta:
            m_smoth = _require_finite_float(
                "benchmark_metadata.ra_filter_smoth",
                meta["ra_filter_smoth"],
                strictly_positive=True,
            )
            if abs(m_smoth - 0.15) > 1e-12:
                raise ValueError(
                    f"benchmark_metadata.ra_filter_smoth={meta['ra_filter_smoth']!r} != 0.15"
                )
        expected_total, expected_head = _count_expected_pom_net_params(
            self.in_channels, self.hidden_channels, self.out_channels, self.num_blocks
        )
        if "closure_parameters" in meta:
            m_tot = _require_int(
                "benchmark_metadata.closure_parameters", meta["closure_parameters"], min_val=1
            )
            if m_tot != expected_total:
                raise ValueError(
                    f"benchmark_metadata.closure_parameters={meta['closure_parameters']!r} "
                    f"conflicts with effective network parameter count {expected_total}"
                )
        if "readout_parameters" in meta:
            m_head = _require_int(
                "benchmark_metadata.readout_parameters", meta["readout_parameters"], min_val=1
            )
            if m_head != expected_head:
                raise ValueError(
                    f"benchmark_metadata.readout_parameters={meta['readout_parameters']!r} "
                    f"conflicts with effective readout parameter count {expected_head}"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("forge_stage2_steps", None)
        return payload

    def save_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        # Sanitize local filesystem paths before persisting public experiment_config.json
        for path_key in ("dataset_root", "output_root", "input_stats_path"):
            raw = str(payload.get(path_key, "")).strip()
            payload[path_key] = Path(raw).name if raw else ""
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


VALID_CONFIG_OVERRIDE_KEYS: frozenset[str] = frozenset(
    f.name for f in fields(ForgePOM3DConfig) if f.name not in NON_OVERRIDABLE_CONFIG_FIELDS
)


def load_pom3d_profile(
    config_path: str | Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Load and validate a structured 3D POM JSON profile (`benchmark_metadata` + `config_overrides`)."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config profile {config_path} must be a JSON object")

    unknown_top = set(raw) - ALLOWED_PROFILE_TOP_KEYS
    if unknown_top:
        raise ValueError(
            f"Unknown top-level keys in {config_path}: {sorted(unknown_top)}; "
            f"expected only {sorted(ALLOWED_PROFILE_TOP_KEYS)}"
        )

    raw_proto = raw.get("protocol", "pure_4h")
    if not isinstance(raw_proto, str) or not raw_proto.strip():
        raise ValueError(f"Invalid 3D POM protocol in {config_path}: {raw_proto!r}")
    protocol = raw_proto.strip()
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(f"Unsupported 3D POM protocol in {config_path}: {protocol!r}")

    metadata = raw.get("benchmark_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"'benchmark_metadata' in {config_path} must be a JSON object")
    unknown_meta = set(metadata) - ALLOWED_BENCHMARK_METADATA_KEYS
    if unknown_meta:
        raise ValueError(
            f"Unknown keys in benchmark_metadata of {config_path}: {sorted(unknown_meta)}"
        )

    overrides = raw.get("config_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"'config_overrides' in {config_path} must be a JSON object")
    unknown_ov = set(overrides) - VALID_CONFIG_OVERRIDE_KEYS
    if unknown_ov:
        raise ValueError(f"Unknown keys in config_overrides of {config_path}: {sorted(unknown_ov)}")

    return protocol, dict(metadata), dict(overrides)


def validate_rollout_horizon_coverage(
    starts: Sequence[int],
    hours: Sequence[int],
    *,
    min_day: int = 1,
) -> tuple[int, ...]:
    """Validate multi-horizon rollout hours and forecast endpoint coverage against [1, 9000] / [0, 8999]."""
    start_days = _require_int_tuple("starts", starts, non_empty=True, min_val=min_day, max_val=9000)
    if isinstance(hours, (str, bytes)) or not isinstance(hours, Sequence) or not hours:
        raise ValueError("integer rollout horizons from 1 to 48 hours required")
    for h_item in hours:
        if (
            isinstance(h_item, bool)
            or not isinstance(h_item, (int, np.integer))
            or not 1 <= int(h_item) <= 48
        ):
            raise ValueError("integer rollout horizons from 1 to 48 hours required")
    sorted_hours = tuple(sorted({int(h) for h in hours}))
    for hour in sorted_hours:
        whole_days, within_day = divmod(hour * 3600, 86400)
        if within_day > 0 and (within_day > 14400 or within_day % 1200 != 0):
            raise ValueError(
                f"Unsupported rollout horizon hour={hour}: intraday reference labels exist only "
                f"in the first 4 hours of each model day (1..4 h, 24 h, 25..28 h, 48 h)"
            )
        for s_day in start_days:
            target_day = s_day + whole_days
            if within_day == 0:
                if not 1 <= target_day <= 9000:
                    raise ValueError(
                        f"Rollout endpoint start_day={s_day} + {hour}h reaches day={target_day} "
                        f"outside daily checkpoint coverage [1, 9000]"
                    )
            elif not 0 <= target_day <= 8999:
                raise ValueError(
                    f"Rollout intraday target start_day={s_day} + {hour}h reaches day={target_day} "
                    f"outside intraday reference coverage [0, 8999]"
                )
    return sorted_hours


POM3DExperimentConfig = ForgePOM3DConfig

__all__ = [
    "ALLOWED_BENCHMARK_METADATA_KEYS",
    "ALLOWED_PROFILE_TOP_KEYS",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DATASET_ROOT_STR",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_OUTPUT_ROOT_STR",
    "ForgePOM3DConfig",
    "NON_OVERRIDABLE_CONFIG_FIELDS",
    "POM3DExperimentConfig",
    "VALID_ABLATION_VARIANTS",
    "VALID_CONFIG_OVERRIDE_KEYS",
    "VALID_PROTOCOLS",
    "load_pom3d_profile",
    "resolve_default_dataset_root",
    "resolve_default_output_root",
    "validate_rollout_horizon_coverage",
]
