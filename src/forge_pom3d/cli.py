"""Pure side-effect-free CLI parser for 3D Double-Gyre POM (`CLI > env > JSON > default`)."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .config import (
    VALID_ABLATION_VARIANTS,
    ForgePOM3DConfig,
    load_pom3d_profile,
)
from .ocean.contracts import resolve_env_var
from .runner import RunnerOptions, normalize_stage_name, run_pipeline

_CLI_TO_CONFIG_FIELD: tuple[tuple[str, str], ...] = (
    ("ablation_variant", "ablation_variant"),
    ("forge_comp_rank", "forge_comp_rank"),
    ("forge_num_directions", "forge_num_directions"),
    ("forge_probe_eps", "forge_probe_eps"),
    ("forge_stage2_probe_amp", "forge_stage2_probe_amp"),
    ("forge_proj_steps", "forge_proj_steps"),
    ("max_window_budget", "max_window_budget"),
    ("forge_rounds", "forge_rounds"),
    ("forge_patience", "forge_patience"),
    ("baseline_rounds", "baseline_rounds"),
    ("baseline_patience", "baseline_patience"),
    ("ad_max_epochs", "ad_max_epochs"),
    ("ad_steps_per_epoch", "ad_steps_per_epoch"),
    ("ad_patience", "ad_patience"),
    ("ppo_lr", "ppo_lr"),
    ("ppo_target_kl", "ppo_target_kl"),
    ("forge_full_gain_bound", "forge_full_gain_bound"),
    ("seed", "seed"),
)


def build_parser() -> argparse.ArgumentParser:
    """Construct pure `ArgumentParser` with `None` defaults so precedence is exact."""
    parser = argparse.ArgumentParser(description="FORGE 3D POM Double-Gyre Pure-4h DDP Pipeline")
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        help="3D POM evaluation protocol (expected: pure_4h)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Structured 3D POM JSON profile path",
    )
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--input-stats", type=str, default=None)
    parser.add_argument("--common-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stages: physics,ad_ref,forge_main,forge_stage1_only,"
            "forge_ablation,emulator,closure_rl,eki,es,ppo,test_4h (or stage0..stage8 aliases)"
        ),
    )
    parser.add_argument(
        "--stage-one-only",
        action="store_true",
        default=False,
        help="Disable Stage II in FORGE main",
    )
    parser.add_argument(
        "--ablation-variant",
        type=str,
        default=None,
        choices=list(VALID_ABLATION_VARIANTS),
        help="Single-factor Stage I/II ablation variant for FORGE 3D POM",
    )
    parser.add_argument(
        "--forge-comp-rank",
        type=int,
        default=None,
        help="Stage I feature-drift SVD cancellation rank",
    )
    parser.add_argument("--forge-num-directions", type=int, default=None)
    parser.add_argument("--forge-probe-eps", type=float, default=None)
    parser.add_argument("--forge-stage2-probe-amp", type=float, default=None)
    parser.add_argument("--forge-proj-steps", type=int, default=None)
    parser.add_argument(
        "--max-window-budget",
        type=float,
        default=None,
        help="Target cumulative window rollout budget cap",
    )
    parser.add_argument("--forge-rounds", type=int, default=None)
    parser.add_argument("--forge-patience", type=int, default=None)
    parser.add_argument("--baseline-rounds", type=int, default=None)
    parser.add_argument("--baseline-patience", type=int, default=None)
    parser.add_argument("--ad-max-epochs", type=int, default=None)
    parser.add_argument("--ad-steps-per-epoch", type=int, default=None)
    parser.add_argument("--ad-patience", type=int, default=None)
    parser.add_argument("--ppo-lr", type=float, default=None)
    parser.add_argument("--ppo-target-kl", type=float, default=None)
    parser.add_argument("--forge-full-gain-bound", type=float, default=None)
    parser.add_argument(
        "--source-sha256",
        type=str,
        default=None,
        help="Expected deterministic forge_pom3d bundle or tree SHA256",
    )
    parser.add_argument(
        "--eval-ckpt",
        type=str,
        default=None,
        help="Frozen checkpoint path for standalone test_4h stage",
    )
    parser.add_argument(
        "--eval-stage-summary",
        type=str,
        default=None,
        help="Source stage summary.json for provenance check in test_4h",
    )
    parser.add_argument(
        "--physics-test-4h-json",
        type=str,
        default=None,
        help="Frozen stage0_physics/test_4h.json for paired Test ID stats",
    )
    parser.add_argument(
        "--eval-out-json",
        type=str,
        default=None,
        help="Custom output JSON path for standalone test_4h",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Run a 2-round smoke check",
    )
    return parser


def parse_config(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[ForgePOM3DConfig, RunnerOptions]:
    """Parse `(ForgePOM3DConfig, RunnerOptions)` with strict precedence `CLI > env > JSON > default`.

    Never mutates `sys.argv` or `os.environ`.
    """
    env = os.environ if environ is None else environ
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Layer 1: Defaults
    cfg_kwargs: dict[str, Any] = {}
    benchmark_metadata: dict[str, Any] = {}
    protocol = "pure_4h"
    stages_raw = "forge_main"

    # Layer 2: Structured JSON profile (`--config` or `FORGE_CONFIG`)
    config_path = args.config
    if config_path is None:
        env_cfg = str(env.get("FORGE_CONFIG", "")).strip()
        if env_cfg:
            config_path = env_cfg
    if config_path:
        json_protocol, benchmark_metadata, json_overrides = load_pom3d_profile(config_path)
        protocol = json_protocol
        cfg_kwargs.update(json_overrides)

    # Layer 3: Environment variables (FORGE_POM3D_*)
    env_protocol = resolve_env_var("FORGE_POM3D_PROTOCOL", "", environ=env).strip()
    env_dataset_root = resolve_env_var("FORGE_POM3D_DATASET_ROOT", "", environ=env).strip()
    env_output_root = resolve_env_var("FORGE_POM3D_OUTPUT_ROOT", "", environ=env).strip()
    env_input_stats = resolve_env_var("FORGE_POM3D_INPUT_STATS", "", environ=env).strip()
    env_stage = resolve_env_var("FORGE_POM3D_STAGE", "", environ=env).strip()

    if env_protocol:
        protocol = env_protocol
    if env_dataset_root:
        cfg_kwargs["dataset_root"] = env_dataset_root
    if env_output_root:
        cfg_kwargs["output_root"] = env_output_root
    if env_input_stats:
        cfg_kwargs["input_stats_path"] = env_input_stats
    if env_stage:
        stages_raw = env_stage

    # Layer 4: Explicit CLI flags (highest precedence)
    if args.protocol is not None:
        protocol = args.protocol.strip()
    if args.dataset_root is not None:
        cfg_kwargs["dataset_root"] = args.dataset_root
    if args.output_root is not None:
        cfg_kwargs["output_root"] = args.output_root
    if args.input_stats is not None:
        cfg_kwargs["input_stats_path"] = args.input_stats
    if args.stages is not None:
        stages_raw = args.stages

    for arg_attr, field_name in _CLI_TO_CONFIG_FIELD:
        val = getattr(args, arg_attr)
        if val is not None:
            cfg_kwargs[field_name] = val

    if args.smoke:
        cfg_kwargs["forge_rounds"] = 2
        cfg_kwargs["baseline_rounds"] = 2
        cfg_kwargs["ad_max_epochs"] = 2
        cfg_kwargs["ad_steps_per_epoch"] = 2

    cfg_kwargs["protocol_mode"] = protocol
    cfg = ForgePOM3DConfig(**cfg_kwargs)
    cfg.validate(benchmark_metadata=benchmark_metadata)

    parsed_stages = tuple(normalize_stage_name(st) for st in stages_raw.split(",") if st.strip())
    if not parsed_stages:
        raise ValueError("No valid stage provided in --stages / FORGE_POM3D_STAGE")
    if "forge_ablation" in parsed_stages and cfg.ablation_variant == "none":
        raise ValueError("--stages forge_ablation requires --ablation-variant != 'none'")

    options = RunnerOptions(
        stages=parsed_stages,
        common_ckpt=args.common_ckpt,
        stage_one_only=bool(args.stage_one_only),
        expected_source_sha256=args.source_sha256,
        eval_ckpt=args.eval_ckpt,
        eval_stage_summary=args.eval_stage_summary,
        physics_test_4h_json=args.physics_test_4h_json,
        eval_out_json=args.eval_out_json,
    )
    return cfg, options


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint parsing config cleanly and invoking `run_pipeline`."""
    cfg, options = parse_config(argv=argv)
    return run_pipeline(cfg, options)


__all__ = ["build_parser", "main", "parse_config"]
