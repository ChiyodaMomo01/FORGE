#!/usr/bin/env python3
"""Read-only held-out Test ID evaluation for frozen 2D neural closure checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


def summarize_parent_aggregation(score_dict: dict[str, Any]) -> dict[str, Any]:
    """Aggregate window MSE by independent held-out parent trajectory."""
    per_window = score_dict["per_window"]
    groups = score_dict["groups"]
    flat_identities = [ident for g in groups for ident in g["identities"]]
    if len(flat_identities) != len(per_window):
        raise ValueError(
            f"per_window length {len(per_window)} != flattened identities {len(flat_identities)}"
        )

    by_parent: dict[str, list[float]] = defaultdict(list)
    for ident, mse_val in zip(flat_identities, per_window, strict=True):
        by_parent[str(ident["parent_id"])].append(float(mse_val))

    parent_means = {pid: float(np.mean(vals)) for pid, vals in sorted(by_parent.items())}
    arr = np.asarray(list(parent_means.values()), dtype=np.float64)
    return {
        "parent_count": len(parent_means),
        "windows_per_parent": int(len(per_window) // max(len(parent_means), 1)),
        "mean_across_parents": float(np.mean(arr)),
        "std_across_parents_ddof1": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min_parent_mse": float(np.min(arr)),
        "max_parent_mse": float(np.max(arr)),
        "per_parent_mse": parent_means,
    }


def _relative_checkpoint_id(ckpt_path: Path, ref_root: Path) -> str:
    try:
        return ckpt_path.resolve().relative_to(ref_root.resolve()).as_posix()
    except ValueError:
        return ckpt_path.name


def setup_isolated_freeze_root(
    freeze_root: Path,
    reference_root: Path,
    case_name: str,
    protocol: str,
    allowed_hashes: list[str],
) -> tuple[str, str]:
    from forge.core.io import file_digest
    from forge.schemas import is_valid_case_protocol

    freeze_root_check = reference_root / "research-freeze.json"
    if freeze_root_check.is_file():
        ref_freeze = json.loads(freeze_root_check.read_text(encoding="utf-8"))
        if not is_valid_case_protocol(ref_freeze.get("protocol"), case_name):
            raise ValueError(
                f"test_id evaluation requires frozen matching protocol in {freeze_root_check.name}: "
                f"{ref_freeze.get('protocol')!r} is invalid for {case_name!r}"
            )

    freeze_root.mkdir(parents=True, exist_ok=True)
    for fname in ("window-manifest.json", "common.pt", "common.json"):
        src = (reference_root / fname).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Missing required reference artifact: {src}")
        dst = freeze_root / fname
        if dst.is_symlink() or dst.exists():
            if not dst.is_symlink() or dst.resolve() != src:
                raise RuntimeError(f"Immutable freeze_root conflict on {dst.name}")
        else:
            dst.symlink_to(src)

    manifest_sha = file_digest(reference_root / "window-manifest.json")
    common_sha = file_digest(reference_root / "common.pt")
    freeze_file = freeze_root / "research-freeze.json"
    merged_hashes = sorted(set(allowed_hashes))
    if freeze_file.is_file():
        existing = json.loads(freeze_file.read_text(encoding="utf-8"))
        if not is_valid_case_protocol(existing.get("protocol"), case_name):
            raise RuntimeError(
                f"Immutable freeze_root conflict in {freeze_file.name} on 'protocol': "
                f"{existing.get('protocol')!r} is invalid for {case_name!r}"
            )
        for key, expected in (
            ("case", case_name),
            ("window_manifest_sha256", manifest_sha),
            ("common_checkpoint_sha256", common_sha),
        ):
            if existing.get(key) != expected:
                raise RuntimeError(
                    f"Immutable freeze_root conflict in {freeze_file.name} on {key!r}: "
                    f"{existing.get(key)!r} != {expected!r}"
                )
        merged_hashes = sorted(set(existing.get("allowed_policy_hashes", [])) | set(allowed_hashes))

    freeze_payload = {
        "case": case_name,
        "protocol": protocol,
        "window_manifest_sha256": manifest_sha,
        "common_checkpoint_sha256": common_sha,
        "allowed_policy_hashes": merged_hashes,
        "frozen_at_unix": time.time(),
        "note": "Isolated read-only Test ID evaluation sandbox.",
    }
    freeze_file.write_text(
        json.dumps(freeze_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_sha, common_sha


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=("cahn_hilliard", "gray_scott", "kolmogorov_ns", "mhd2d"),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON benchmark profile path (with optional recipe_overrides)",
    )
    parser.add_argument(
        "--reference-root",
        required=True,
        help="Prepared reference run directory containing window-manifest.json and common.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to frozen selected.pt checkpoint (defaults to <reference-root>/common.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=("network", "pure_physics"),
        default="network",
    )
    parser.add_argument("--actor-fp64", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    if args.world_size < 1:
        raise ValueError("--world-size must be >= 1")

    from forge.core.io import file_digest, save
    from forge.learning.client import ShardedClient
    from forge.learning.common import array, digest, load_model
    from forge.learning.numeric import configure
    from forge.learning.schema import RECIPE_BY_CASE, load_recipe_profile

    ref_root = Path(args.reference_root).resolve()
    out_json = Path(args.output_json).resolve()
    freeze_root = out_json.parent / f"_freeze_sandbox_{out_json.stem}"
    worker_root = out_json.parent / f"_eval_workers_{out_json.stem}"

    base_recipe = (
        load_recipe_profile(args.config, case_name=args.case)
        if args.config
        else RECIPE_BY_CASE[args.case]
    )
    recipe = replace(base_recipe, actor_fp64=bool(args.actor_fp64))
    configure(recipe)

    common_ckpt = (ref_root / "common.pt").resolve()
    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else common_ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    common_model, _ = load_model(common_ckpt, device="cpu")
    common_hash = digest(array(common_model))

    model, meta = load_model(ckpt_path, device="cpu")
    if model.case_name != args.case:
        raise ValueError(
            f"Checkpoint case mismatch: checkpoint is {model.case_name!r}, requested {args.case!r}"
        )
    theta = array(model)
    policy_hash = digest(theta)

    manifest_sha, common_sha = setup_isolated_freeze_root(
        freeze_root=freeze_root,
        reference_root=ref_root,
        case_name=args.case,
        protocol=recipe.protocol,
        allowed_hashes=[policy_hash, common_hash],
    )

    t0 = time.monotonic()
    with ShardedClient(
        freeze_root,
        worker_root,
        args.device,
        recipe,
        checkpoint_path=freeze_root / "common.pt",
        world_size=args.world_size,
    ) as client:
        score = client.score(
            theta,
            "test_id",
            None,
            mode=args.mode,
            label=f"test-id-{args.case}-{args.mode}",
        )
    elapsed = time.monotonic() - t0

    raw_failures = score.get("failures", [])
    failure_count = len(raw_failures) if isinstance(raw_failures, list) else int(raw_failures)
    parent_agg = summarize_parent_aggregation(score)

    summary = {
        "case": args.case,
        "protocol": recipe.protocol,
        "mode": args.mode,
        "actor_fp64": bool(args.actor_fp64),
        "checkpoint": _relative_checkpoint_id(ckpt_path, ref_root),
        "checkpoint_sha256": file_digest(ckpt_path),
        "policy_hash": policy_hash,
        "window_manifest_sha256": manifest_sha,
        "common_checkpoint_sha256": common_sha,
        "checkpoint_meta": {
            k: meta.get(k)
            for k in ("method", "selected_round", "selected_epoch", "validation_mse")
            if k in meta
        },
        "mean_window_mse": float(score["mean_mse"]),
        "uncapped_mean_mse": score.get("uncapped_mean_mse"),
        "failures": failure_count,
        "nonfinite_outputs": int(score.get("uncapped_nonfinite_outputs", 0)),
        "window_attempts": int(score["cost"]["window_attempts"]),
        "parent_aggregation": parent_agg,
        "wall_seconds": elapsed,
    }
    save(out_json, summary)
    print(
        f"[TEST ID | {args.case} | {args.mode}] mean_mse={summary['mean_window_mse']:.6e} "
        f"| parent_mean={parent_agg['mean_across_parents']:.6e} "
        f"| failures={summary['failures']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
