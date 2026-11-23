"""Unified training and evaluation CLI for all four 2D cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

from .core.contracts import (
    RECIPE_BY_CASE,
    apply_validated_recipe_overrides,
    configure,
    load_recipe_profile,
    validate_profile_data,
)
from .core.io import save
from .execution import (
    compute_effective_config_sha256,
    compute_source_bundle_sha256,
    compute_source_tree_sha256,
)
from .learning.ablations import add_arguments, apply_arguments, variant
from .stages import execute

METHODS = (
    "pure_physics",
    "common",
    "ES",
    "EKI",
    "PPO_global",
    "PPO_local",
    "emulator",
    "closure_rl",
    "main",
    "AD",
)

ACTIVE_2D_CASES = tuple(RECIPE_BY_CASE)
ALL_2D_METHODS = (
    "prepare",
    "physics_start",
    "pretrain",
    "freeze_common",
    "validate_gate",
    *METHODS,
)


def _sanitize_relative_label(target: str | Path, anchor: Path) -> str:
    resolved_target = Path(target).resolve()
    resolved_anchor = anchor.resolve()
    try:
        return resolved_target.relative_to(resolved_anchor).as_posix()
    except ValueError:
        return resolved_target.name


def build_normalized_argv(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Resolve CLI > environment > JSON profile > default precedence without mutating globals."""
    raw = list(sys.argv[1:] if argv is None else argv)
    environ = os.environ if env is None else env

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    pre.add_argument("--case", default="")
    pre.add_argument("--method", default="")
    known, rest = pre.parse_known_args(raw)

    config_path = known.config or str(environ.get("FORGE_CONFIG", "")).strip()
    env_case = str(environ.get("FORGE_CASE", "")).strip()
    env_method = str(environ.get("FORGE_METHOD", "")).strip()

    profile: dict = {}
    profile_case = ""
    if config_path:
        raw_profile = json.loads(Path(config_path).read_text(encoding="utf-8"))
        explicit_case = known.case or env_case or None
        if explicit_case is not None and explicit_case not in ACTIVE_2D_CASES:
            raise ValueError(f"Unsupported 2D benchmark case: {explicit_case!r}")
        profile_case, profile = validate_profile_data(raw_profile, case_name=explicit_case)

    case_name = known.case or env_case or profile_case
    method = known.method or env_method or str(profile.get("default_method") or "").strip()
    if not case_name:
        raise ValueError("--case is required (via CLI, FORGE_CASE, or --config JSON)")
    if not method:
        raise ValueError("--method is required")

    if case_name not in ACTIVE_2D_CASES:
        raise ValueError(f"Unsupported 2D benchmark case: {case_name!r}")
    if method not in ALL_2D_METHODS:
        raise ValueError(f"Unsupported 2D stage/method: {method!r}")

    present_flags = {tok.split("=", 1)[0] for tok in rest if tok.startswith("--")}
    forwarded: list[str] = ["--case", case_name, "--method", method]
    if config_path and "--config" not in present_flags:
        forwarded.extend(["--config", config_path])

    env_and_json_defaults = (
        ("--output", "FORGE_OUTPUT_ROOT", "output_root"),
        ("--pairs", "FORGE_PAIRS_ROOT", "pairs_root"),
        ("--original", "FORGE_ORIGINAL_ROOT", "original_root"),
        ("--reference", "FORGE_REFERENCE_ROOT", "reference_root"),
        ("--device", "FORGE_DEVICE", "device"),
        ("--source-sha256", "FORGE_SOURCE_SHA256", "source_sha256"),
    )
    for flag, env_key, json_key in env_and_json_defaults:
        if flag in present_flags:
            continue
        env_val = str(environ.get(env_key, "")).strip()
        if env_val:
            forwarded.extend([flag, env_val])
        elif json_key in profile and profile[json_key] not in (None, ""):
            forwarded.extend([flag, str(profile[json_key])])

    recipe_overrides = profile.get("recipe_overrides", {})
    if (
        isinstance(recipe_overrides, dict)
        and recipe_overrides
        and "--recipe-overrides-json" not in present_flags
    ):
        forwarded.extend(["--recipe-overrides-json", json.dumps(recipe_overrides)])

    cli_defaults = profile.get("cli_defaults", {})
    if isinstance(cli_defaults, dict) and method == "main":
        for flag, val in cli_defaults.items():
            if flag in present_flags:
                continue
            if isinstance(val, bool):
                if val:
                    forwarded.append(flag)
            elif val is not None:
                forwarded.extend([flag, str(val)])

    forwarded.extend(rest)
    return forwarded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=ALL_2D_METHODS,
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=ACTIVE_2D_CASES,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--original", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference")
    parser.add_argument("--source-sha256", default="forge-release")
    parser.add_argument("--recipe-overrides-json", default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--body-basis-mode", choices=("structured", "dense"))
    parser.add_argument("--body-compensation-rank", type=int)
    parser.add_argument(
        "--stage-one-trust-mode",
        choices=("body_only", "uniform_ray", "conditional_resolve"),
    )
    parser.add_argument("--damping-mode", choices=("isotropic", "response_scaled"))
    parser.add_argument("--head-weight-penalty-eta", type=float)
    parser.add_argument("--resume-from-attempt", default=None)
    parser.add_argument(
        "--folder-name",
        default=None,
        help="Optional output subfolder override.",
    )
    parser.add_argument(
        "--target-window-budget",
        type=int,
        default=0,
        help="Target window_attempts ceiling for equal-budget Stage-I-only ablation.",
    )
    parser.add_argument(
        "--ppo-lr",
        type=float,
        default=None,
        help="Optional learning rate override for --method PPO_global or PPO_local.",
    )
    parser.add_argument(
        "--ad-lr",
        type=float,
        default=None,
        help="Optional learning rate override for --method AD.",
    )
    parser.add_argument(
        "--ad-epochs",
        type=int,
        default=None,
        help="Optional planned epoch cap override for --method AD.",
    )
    parser.add_argument(
        "--ad-tag",
        default=None,
        help="Write AD outputs to AD-<tag>; existing AD/ folder is never reused.",
    )
    add_arguments(parser)
    parser.add_argument("--config", help="JSON profile (CLI > environment > JSON > defaults)")
    return parser


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if any(flag in raw for flag in ("-h", "--help")):
        parser.parse_args(raw)
    args = parser.parse_args(build_normalized_argv(raw))

    if args.method in ("prepare", "physics_start", "pretrain") and not args.pairs:
        raise ValueError(f"--pairs (or FORGE_PAIRS_ROOT) is required for --method {args.method}")
    if args.method == "prepare" and not args.original:
        raise ValueError("--original (or FORGE_ORIGINAL_ROOT) is required for --method prepare")
    if args.resume_from_attempt is not None and args.method != "main":
        raise ValueError("--resume-from-attempt is only supported for --method main")
    if args.target_window_budget < 0:
        raise ValueError("--target-window-budget must be non-negative")
    if args.folder_name is not None:
        if args.ad_tag is not None:
            raise ValueError("--folder-name cannot be combined with --ad-tag")
        if args.folder_name in (".", "..") or not re.fullmatch(
            r"[A-Za-z0-9._-]+", args.folder_name
        ):
            raise ValueError(
                "--folder-name must be a non-empty alphanumeric/._- identifier without path separators"
            )
    if args.ppo_lr is not None:
        if args.method not in ("PPO_global", "PPO_local"):
            raise ValueError("--ppo-lr is only valid for --method PPO_global or PPO_local")
        if not math.isfinite(args.ppo_lr) or args.ppo_lr <= 0.0:
            raise ValueError("--ppo-lr must be positive and finite")
    if (
        args.ad_lr is not None or args.ad_epochs is not None or args.ad_tag is not None
    ) and args.method != "AD":
        raise ValueError("--ad-lr, --ad-epochs, and --ad-tag are only valid for --method AD")
    if args.ad_lr is not None and (not math.isfinite(args.ad_lr) or args.ad_lr <= 0.0):
        raise ValueError("--ad-lr must be positive and finite")
    if args.ad_epochs is not None and args.ad_epochs <= 0:
        raise ValueError("--ad-epochs must be positive")
    if args.ad_tag is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", args.ad_tag):
        raise ValueError(
            "--ad-tag must be a non-empty alphanumeric/._- identifier without path separators"
        )

    root = Path(args.output)
    folder_name = (
        f"AD-{args.ad_tag}"
        if args.ad_tag
        else (args.folder_name if args.folder_name else args.method)
    )
    args.resolved_folder_name = folder_name
    folder = root / folder_name

    recipe = (
        load_recipe_profile(args.config, case_name=args.case)
        if args.config
        else RECIPE_BY_CASE[args.case]
    )
    if args.recipe_overrides_json:
        raw_ov = json.loads(args.recipe_overrides_json)
        recipe = apply_validated_recipe_overrides(recipe, raw_ov)
    if args.ppo_lr is not None:
        recipe = replace(recipe, ppo_lr=float(args.ppo_lr))
    if args.ad_lr is not None:
        recipe = replace(recipe, ad_lr=float(args.ad_lr))
    if args.ad_epochs is not None:
        recipe = replace(recipe, ad_epochs=int(args.ad_epochs))
    if args.stage_one_only:
        s1_win_per_round = max(35 * 4 * recipe.training_groups_per_round, 1)
        default_s1_rounds = (
            max(
                recipe.fine_rounds + 20,
                math.ceil(args.target_window_budget / s1_win_per_round) + 8,
            )
            if args.target_window_budget > 0
            else recipe.fine_rounds
        )
        recipe = replace(
            recipe,
            disable_stage_two=True,
            fine_rounds=default_s1_rounds,
            ignore_early_stop=(args.target_window_budget > 0),
            target_window_budget=int(args.target_window_budget),
        )
    stage_one_keys = (
        "body_basis_mode",
        "body_compensation_rank",
        "stage_one_trust_mode",
        "damping_mode",
        "head_weight_penalty_eta",
    )
    stage_one = {
        key: getattr(args, key) for key in stage_one_keys if getattr(args, key) is not None
    }
    if stage_one and args.method != "main":
        raise ValueError("Stage I options require --method main")
    recipe = replace(recipe, **stage_one)
    if args.max_rounds is not None:
        if args.max_rounds <= 0:
            raise ValueError("--max-rounds must be positive")
        recipe = replace(recipe, fine_rounds=args.max_rounds)
    if args.method in ("PPO_global", "PPO_local", "closure_rl"):
        recipe = replace(recipe, actor_fp64=True)
    recipe = apply_arguments(recipe, args)

    distributed = None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from .learning import distributed as execution
        from .learning import distributed_jobs as jobs

        torch.set_num_threads(1)
        torch.manual_seed(0)
        configure(recipe)
        if args.method == "AD":
            from .privileged import distributed_trajectory as reference

            jobs.HANDLERS["ad"] = lambda eng, payload: reference.train(eng, **payload)
        if args.method == "pretrain":
            from .paired import pretrain as regression

            jobs.HANDLERS["pretrain"] = lambda eng, payload: regression.train(
                **{**payload, "device": str(eng.device), "distributed": True}
            )
        execution.set_handlers(jobs.HANDLERS)
        distributed = execution.Engine(folder)
        args.device = str(distributed.device)
        if execution.rank() != 0:
            try:
                distributed.serve()
            finally:
                distributed.close(notify=False)
            return
    elif args.device.startswith("cuda") and int(os.environ.get("FORGE_DDP_WORLD_SIZE", "1")) > 1:
        raise RuntimeError("multi-GPU stages require torchrun; DataParallel fallback is forbidden")

    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "status.json").exists():
        raise FileExistsError(
            "attempt already exists; specify a fresh output directory or resume explicitly"
        )
    torch.set_num_threads(1)
    torch.manual_seed(0)
    configure(recipe)
    gpu = args.device.startswith("cuda")
    if gpu:
        if distributed is None:
            raw_mem_frac = os.environ.get("FORGE_CUDA_MEMORY_FRACTION", "").strip()
            if raw_mem_frac:
                torch.cuda.set_per_process_memory_fraction(float(raw_mem_frac))
        torch.cuda.reset_peak_memory_stats()

    source_tree_sha = compute_source_tree_sha256()
    effective_cfg_sha = compute_effective_config_sha256(recipe, method=args.method)
    source_bundle_sha = compute_source_bundle_sha256(source_tree_sha, effective_cfg_sha)

    begun = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    status = {
        "status": "running",
        "pid": os.getpid(),
        "method": args.method,
        "case": args.case,
        "source_sha256": args.source_sha256,
        "source_tree_sha256": source_tree_sha,
        "effective_config_sha256": effective_cfg_sha,
        "source_bundle_sha256": source_bundle_sha,
        "started_unix": time.time(),
        "recipe": recipe.dictionary(),
        "method_variant": variant(recipe),
        "ad_tag": args.ad_tag,
        "folder_name": folder_name,
        "resumed_from_attempt": (
            _sanitize_relative_label(args.resume_from_attempt, root)
            if args.resume_from_attempt
            else None
        ),
    }
    if shutil.which("git") and (Path.cwd() / ".git").exists():
        status["checkout_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        status["checkout_source_sha256"] = hashlib.sha256(
            subprocess.check_output(["git", "archive", "HEAD"])
        ).hexdigest()
    else:
        status["checkout_revision"] = os.environ.get("FORGE_GIT_REVISION", "installed-package")
        status["checkout_source_sha256"] = (
            source_tree_sha if args.source_sha256 == "forge-release" else args.source_sha256
        )
    save(folder / "status.json", status)
    try:
        result = execute(args, recipe)
        if args.method in (
            "prepare",
            "physics_start",
            "freeze_common",
            "validate_gate",
        ):
            save(folder / "result.json", result)
        status["status"] = "complete"
    except BaseException as error:
        status.update(
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
        )
        traceback.print_exc()
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF)
        desc = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(
            folder / "resources.json",
            {
                "wall_seconds": time.monotonic() - begun,
                "learner_cpu_seconds": after.ru_utime
                + after.ru_stime
                - before.ru_utime
                - before.ru_stime,
                "waited_descendant_cpu_seconds": desc.ru_utime
                + desc.ru_stime
                - children.ru_utime
                - children.ru_stime,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if gpu else None,
                "scope": "stage process plus waited descendants CPU once; query/fit ledgers are subsets",
            },
        )
        status["finished_unix"] = time.time()
        save(folder / "status.json", status)
        if distributed is not None:
            if status.get("status") == "complete":
                distributed.close()
            else:
                for value in distributed.state.values():
                    if hasattr(value, "close"):
                        value.close()


if __name__ == "__main__":
    main()
