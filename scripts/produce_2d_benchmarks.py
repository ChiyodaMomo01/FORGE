#!/usr/bin/env python3
"""Produce high-resolution 2D reference trajectories, replay actual-G pairs, and run restriction audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.collect.audit import audit
from forge.collect.replay import collect_case
from forge.config import load_campaign, parent_tasks
from forge.paired.data import EXPECTED_SPLITS
from forge.production import produce_parent


def _resolve_cuda_indices(device: str, replay_devices: str) -> str:
    if replay_devices.strip():
        return replay_devices.strip()
    if device.startswith("cuda"):
        parts = device.split(":", 1)
        return parts[1] if len(parts) == 2 and parts[1].strip() else "0"
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-config",
        default=None,
        help="Path to configs/benchmarks_reaction_2d.json or configs/benchmarks_turbulence_2d.json",
    )
    parser.add_argument(
        "--original-root",
        "--output-root",
        dest="original_root",
        required=True,
        help="Root directory for generated reference trajectories (<original-root>/<case>/<split>/...)",
    )
    parser.add_argument(
        "--pairs-root",
        default=None,
        help="Root directory for actual-G replay collection and restriction audit (<pairs-root>/<case>/...)",
    )
    parser.add_argument(
        "--case",
        choices=("cahn_hilliard", "gray_scott", "kolmogorov_ns", "mhd2d"),
        default=None,
        help="Optional single-case filter",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test_id"),
        default=None,
        help="Optional split filter (when producing partial subsets without full-case replay)",
    )
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--replay-workers",
        type=int,
        default=8,
        help="Worker count for actual-G replay collection (use 1 for single-process CPU execution)",
    )
    parser.add_argument(
        "--replay-devices",
        default="",
        help="Optional comma-separated CUDA device indices (e.g. '0,1,2,3') for multi-GPU HR replay",
    )
    parser.add_argument(
        "--source-revision",
        default="forge-release",
        help="Provenance revision tag recorded in collection.json and complete.json",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Only run HR/LR parent trajectory production without replay collection or audit",
    )
    parser.add_argument(
        "--allow-custom-splits",
        action="store_true",
        help="Allow non-canonical positive multiples-of-4 split counts defined in --campaign-config (for miniature test fixtures)",
    )
    parser.add_argument(
        "--audit-pairs-root",
        default=None,
        help="Run the restriction and F-commutation audit on an existing --audit-pairs-root without generating",
    )
    args = parser.parse_args(argv)

    orig_root = Path(args.original_root).resolve()

    if args.audit_pairs_root:
        if not args.case:
            raise ValueError("--case is required when running --audit-pairs-root")
        result = audit(
            Path(args.audit_pairs_root).resolve(),
            args.case,
            original_root=orig_root,
        )
        print(
            f"[AUDIT PASSED] case={args.case} schema={result.get('schema')}",
            flush=True,
        )
        return

    if not args.campaign_config:
        raise ValueError("--campaign-config is required unless --audit-pairs-root is used")

    config = load_campaign(args.campaign_config)
    tasks = [
        t
        for t in parent_tasks(config)
        if t["case"] in ("cahn_hilliard", "gray_scott", "kolmogorov_ns", "mhd2d")
        and (args.case is None or t["case"] == args.case)
        and (args.split is None or t["split"] == args.split)
    ]
    if args.max_parents is not None:
        tasks = tasks[: args.max_parents]

    if not args.skip_replay and args.pairs_root is not None:
        if args.split is not None or args.max_parents is not None:
            raise ValueError(
                "Full actual-G replay collection and restriction audit require all splits of each selected case "
                "(omit --split and --max-parents, or pass --skip-replay during partial parent generation)."
            )
        cfg_splits = {k: int(v) for k, v in config["splits"].items()}
        for case_name in sorted({t["case"] for t in tasks}):
            canonical = dict(EXPECTED_SPLITS[case_name])
            if cfg_splits != canonical and not args.allow_custom_splits:
                raise ValueError(
                    f"Campaign split counts {cfg_splits} differ from canonical {canonical} for {case_name}; "
                    "pass --allow-custom-splits only for non-benchmark test fixtures."
                )

    orig_root.mkdir(parents=True, exist_ok=True)
    for idx, task in enumerate(tasks, 1):
        done_marker = orig_root / task["case"] / task["split"] / f"{task['parent']}.json"
        if (
            done_marker.is_file()
            and json.loads(done_marker.read_text(encoding="utf-8")).get("status") == "complete"
        ):
            print(
                f"[{idx}/{len(tasks)}] SKIP completed {task['case']} {task['split']} parent={task['parent']}",
                flush=True,
            )
            continue
        print(
            f"[{idx}/{len(tasks)}] Producing {task['case']} {task['split']} parent={task['parent']} on {args.device}",
            flush=True,
        )
        produce_parent(task, orig_root, device=args.device)

    if args.skip_replay or args.pairs_root is None:
        return

    if args.split is not None or args.max_parents is not None:
        raise ValueError(
            "Full actual-G replay collection and restriction audit require all splits of each selected case "
            "(omit --split and --max-parents, or pass --skip-replay during partial parent generation)."
        )

    pairs_root = Path(args.pairs_root).resolve()
    pairs_root.mkdir(parents=True, exist_ok=True)
    selected_cases = sorted({t["case"] for t in tasks})
    replay_devs = _resolve_cuda_indices(args.device, args.replay_devices)
    cfg_splits = {k: int(v) for k, v in config["splits"].items()}

    for case_name in selected_cases:
        canonical = dict(EXPECTED_SPLITS[case_name])
        if cfg_splits != canonical and not args.allow_custom_splits:
            raise ValueError(
                f"Campaign split counts {cfg_splits} differ from canonical {canonical} for {case_name}; "
                "pass --allow-custom-splits only for non-benchmark test fixtures."
            )
        expected_splits = None if cfg_splits == canonical else cfg_splits
        print(
            f"[REPLAY] Collecting actual-G pairs for {case_name} -> {pairs_root / case_name}",
            flush=True,
        )
        collect_case(
            orig_root,
            pairs_root,
            case_name,
            workers=args.replay_workers,
            devices=replay_devs,
            source_revision=args.source_revision,
            expected_splits=expected_splits,
        )
        print(
            f"[AUDIT] Verifying restriction and F-commutation for {case_name}",
            flush=True,
        )
        info = audit(pairs_root, case_name, original_root=orig_root)
        print(
            f"[AUDIT PASSED] case={case_name} schema={info.get('schema')} lr_grid={info.get('lr_grid')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
