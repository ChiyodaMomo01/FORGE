"""Stage dispatcher and DDP execution orchestrator for 3D Double-Gyre POM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import compute_source_digests, save_atomic_json, sha256_file
from .baselines.ad_reference import run_ad_reference
from .baselines.closure_rl import run_closure_rl_baseline
from .baselines.eki import run_eki_baseline
from .baselines.emulator import run_emulator_baseline
from .baselines.es import run_es_baseline
from .baselines.physics import run_physics_baseline
from .baselines.ppo import run_ppo_baseline
from .config import ForgePOM3DConfig
from .distributed import cleanup_ddp, is_main_process, setup_ddp
from .evaluation import run_standalone_test_4h
from .physics_env import BlackBoxProvider3D
from .sampling import (
    build_train_window_pool,
    build_val_windows,
    ensure_common_zero_ckpt,
)
from .training.trainer import run_forge_main_4h

STAGE_ALIASES: dict[str, str] = {
    "physics": "physics",
    "stage0_physics": "physics",
    "forge_main": "forge_main",
    "stage2_forge_main": "forge_main",
    "forge_stage1_only": "forge_stage1_only",
    "stage1_only": "forge_stage1_only",
    "stage2_forge_stage1_only": "forge_stage1_only",
    "forge_ablation": "forge_ablation",
    "stage2_forge_ablation": "forge_ablation",
    "es": "es",
    "stage3_es": "es",
    "eki": "eki",
    "stage4_eki": "eki",
    "ppo": "ppo",
    "stage5_ppo": "ppo",
    "emulator": "emulator",
    "stage6_emulator": "emulator",
    "closure_rl": "closure_rl",
    "stage7_closure_rl": "closure_rl",
    "ad_ref": "ad_ref",
    "stage8_ad_ref": "ad_ref",
    "test_4h": "test_4h",
    "eval_test_4h": "test_4h",
}


@dataclass(frozen=True)
class RunnerOptions:
    """Execution options passed from CLI to `run_pipeline`."""

    stages: tuple[str, ...] = ("forge_main",)
    common_ckpt: str | None = None
    stage_one_only: bool = False
    expected_source_sha256: str | None = None
    eval_ckpt: str | None = None
    eval_stage_summary: str | None = None
    physics_test_4h_json: str | None = None
    eval_out_json: str | None = None


def normalize_stage_name(stage: str) -> str:
    """Resolve stage name or alias to canonical runner dispatch key."""
    token = str(stage).strip()
    if token not in STAGE_ALIASES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_ALIASES)}")
    return STAGE_ALIASES[token]


def run_pipeline(cfg: ForgePOM3DConfig, options: RunnerOptions) -> dict[str, Any]:
    """Execute requested 3D POM training/evaluation stages under DDP."""
    if not isinstance(cfg, ForgePOM3DConfig):
        raise ValueError(f"cfg must be a ForgePOM3DConfig instance, got {type(cfg).__name__}")
    cfg.validate()
    if not isinstance(options.stage_one_only, bool):
        raise ValueError("RunnerOptions boolean flags must be strict booleans")
    normalized_stages = tuple(normalize_stage_name(st) for st in options.stages)
    if not normalized_stages:
        raise ValueError("At least one stage must be specified")
    if "forge_ablation" in normalized_stages and cfg.ablation_variant == "none":
        raise ValueError("--stages forge_ablation requires --ablation-variant != 'none'")

    compute_source_digests(cfg)
    if options.expected_source_sha256 is not None:
        expected = options.expected_source_sha256.strip()
        if expected not in (cfg.source_bundle_sha256, cfg.source_tree_sha256):
            raise RuntimeError(
                f"source_sha256 mismatch: expected={expected} "
                f"bundle={cfg.source_bundle_sha256} tree={cfg.source_tree_sha256}"
            )

    rank, world_size, _local_rank, device = setup_ddp()
    try:
        env = BlackBoxProvider3D(cfg, device)
        train_pool = build_train_window_pool(cfg)
        val_windows = build_val_windows(cfg)

        explicit_common = Path(options.common_ckpt) if options.common_ckpt is not None else None
        common_ckpt = ensure_common_zero_ckpt(cfg, rank, device, explicit_ckpt=explicit_common)
        cfg.common_hash = sha256_file(common_ckpt)

        out_root_path = Path(cfg.output_root)
        if is_main_process(rank):
            cfg.save_json(out_root_path / "experiment_config.json")
            print(
                f"=== FORGE 3D POM Pure-4h DDP Pipeline | world_size={world_size} | seed={cfg.seed} "
                f"| src={cfg.source_version} ({cfg.source_bundle_sha256[:16]}) "
                f"| common_sha256={cfg.common_hash[:12]} | run={out_root_path.name} ===",
                flush=True,
            )

        results: dict[str, Any] = {}
        budget_cap = cfg.max_window_budget

        for st in normalized_stages:
            if st == "physics":
                results["stage0_physics"] = run_physics_baseline(
                    cfg, env, val_windows, rank, world_size, device
                )
            elif st == "ad_ref":
                results["stage8_ad_ref"] = run_ad_reference(
                    cfg, env, common_ckpt, train_pool, val_windows, rank, world_size, device
                )
            elif st == "forge_stage1_only":
                results["stage2_forge_stage1_only"] = run_forge_main_4h(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    stage_one_only=True,
                    ablation_variant="none",
                    max_window_budget=budget_cap,
                )
            elif st == "forge_ablation":
                abl_key = f"stage2_forge_ablation_{cfg.ablation_variant}"
                results[abl_key] = run_forge_main_4h(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    stage_one_only=False,
                    ablation_variant=cfg.ablation_variant,
                    max_window_budget=budget_cap,
                )
            elif st == "forge_main":
                main_key = (
                    "stage2_forge_stage1_only"
                    if bool(options.stage_one_only)
                    else (
                        "stage2_forge_main"
                        if cfg.ablation_variant == "none"
                        else f"stage2_forge_ablation_{cfg.ablation_variant}"
                    )
                )
                results[main_key] = run_forge_main_4h(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    stage_one_only=bool(options.stage_one_only),
                    ablation_variant=cfg.ablation_variant,
                    max_window_budget=budget_cap,
                )
            elif st == "es":
                results["stage3_es"] = run_es_baseline(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    max_window_budget=budget_cap,
                )
            elif st == "eki":
                results["stage4_eki"] = run_eki_baseline(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    max_window_budget=budget_cap,
                )
            elif st == "ppo":
                results["stage5_ppo"] = run_ppo_baseline(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    max_window_budget=budget_cap,
                )
            elif st == "emulator":
                results["stage6_emulator"] = run_emulator_baseline(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    max_window_budget=budget_cap,
                )
            elif st == "closure_rl":
                results["stage7_closure_rl"] = run_closure_rl_baseline(
                    cfg,
                    env,
                    common_ckpt,
                    train_pool,
                    val_windows,
                    rank,
                    world_size,
                    device,
                    max_window_budget=budget_cap,
                )
            elif st == "test_4h":
                results["test_4h"] = run_standalone_test_4h(
                    cfg,
                    env,
                    common_ckpt,
                    rank,
                    world_size,
                    device,
                    eval_ckpt_arg=options.eval_ckpt,
                    eval_stage_summary_arg=options.eval_stage_summary,
                    physics_test_4h_json_arg=options.physics_test_4h_json,
                    eval_out_json_arg=options.eval_out_json,
                )
            else:
                raise ValueError(f"Unhandled normalized stage {st!r}")

        if is_main_process(rank):
            save_atomic_json(out_root_path / "pipeline_results.json", results)
            print(
                "=== FORGE 3D POM Pure-4h DDP Pipeline Completed Successfully ===",
                flush=True,
            )
        return results
    finally:
        cleanup_ddp()


__all__ = [
    "RunnerOptions",
    "STAGE_ALIASES",
    "normalize_stage_name",
    "run_pipeline",
]
