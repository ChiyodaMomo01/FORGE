#!/usr/bin/env python3
"""Evaluate frozen 3D POM closure checkpoints on 1-to-48-hour closed-loop rollouts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("pure_4h",),
        default="pure_4h",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional structured 3D POM JSON profile path",
    )
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--input-stats", default="")
    parser.add_argument("--mode", choices=("pure_physics", "network"), default="network")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--starts-preset",
        choices=("full48", "continuity8"),
        default="full48",
        help="Test start-day preset when --split test (default: full48, days 8000..8940 stride 20)",
    )
    parser.add_argument(
        "--starts",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit start days overriding the split default",
    )
    parser.add_argument("--hours", type=int, nargs="+", default=[1, 2, 4, 24, 48])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)

    from forge_pom3d.artifacts import (
        compute_source_digests,
        relativize_artifact_path,
        sha256_file,
    )
    from forge_pom3d.config import (
        ForgePOM3DConfig,
        load_pom3d_profile,
        resolve_default_dataset_root,
        validate_rollout_horizon_coverage,
    )
    from forge_pom3d.network import ForgePOMNet
    from forge_pom3d.physics_env import BlackBoxProvider3D
    from forge_pom3d.sampling import build_test_id_48_windows, build_test_id_continuity8_windows

    out_path = Path(args.output_json).resolve()
    cfg_kwargs: dict[str, object] = {}
    benchmark_metadata: dict[str, object] = {}
    if args.config:
        json_proto, benchmark_metadata, json_ov = load_pom3d_profile(args.config)
        if json_proto != args.protocol:
            raise ValueError(f"Profile protocol {json_proto!r} != --protocol {args.protocol!r}")
        cfg_kwargs.update(json_ov)

    ds_arg = str(args.dataset_root).strip()
    if ds_arg:
        cfg_kwargs["dataset_root"] = ds_arg
    elif "dataset_root" not in cfg_kwargs:
        cfg_kwargs["dataset_root"] = str(resolve_default_dataset_root())
    if str(args.input_stats).strip():
        cfg_kwargs["input_stats_path"] = str(args.input_stats).strip()
    cfg_kwargs["output_root"] = str(out_path.parent / "_eval_cfg_stub")
    cfg_kwargs["protocol_mode"] = args.protocol

    cfg = ForgePOM3DConfig(**cfg_kwargs)  # type: ignore[arg-type]
    cfg.validate(benchmark_metadata=benchmark_metadata)  # type: ignore[arg-type]

    if args.split == "validation":
        member = int(cfg.val_members[0])
        starts = (
            [int(d) for d in args.starts]
            if args.starts is not None
            else list(range(cfg.val_start_day, cfg.val_end_day, 120))
        )
    else:
        member = int(cfg.eval_member)
        if args.starts is not None:
            starts = [int(d) for d in args.starts]
        elif args.starts_preset == "continuity8":
            starts = [int(d) for _, d in build_test_id_continuity8_windows(cfg)]
        else:
            starts = [int(d) for _, d in build_test_id_48_windows(cfg)]

    validate_rollout_horizon_coverage(starts, args.hours, min_day=max(1, cfg.spinup_days))
    if args.mode == "network" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --mode network")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    digests = compute_source_digests(cfg)
    env = BlackBoxProvider3D(cfg, device=device)

    ckpt_rel: str | None = None
    if args.mode == "pure_physics":
        policy_fn = None
        ckpt_sha = None
    else:
        ckpt_path = Path(args.checkpoint).resolve()
        ckpt_sha = sha256_file(ckpt_path)
        ckpt_rel = relativize_artifact_path(ckpt_path, out_path.parent)
        model = ForgePOMNet(
            cfg.in_channels, cfg.hidden_channels, cfg.out_channels, cfg.num_blocks
        ).to(device)
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(payload, dict):
            for key in ("model_state_dict", "model", "state_dict"):
                if key in payload and isinstance(payload[key], dict):
                    state_dict = payload[key]
                    break
            else:
                state_dict = payload
        else:
            state_dict = payload
        model.load_state_dict(state_dict)
        model.eval()

        def _policy_fn(obs: torch.Tensor, _step_k: int) -> torch.Tensor:
            return model(obs)

        policy_fn = _policy_fn

    t0 = time.monotonic()
    records = [
        env.evaluate_hourly_rollout(member, int(start_day), args.hours, policy_fn)
        for start_day in starts
    ]

    summary = {
        "protocol": args.protocol,
        "split": args.split,
        "member": member,
        "num_starts": len(starts),
        "start_days": starts,
        "hours": sorted(set(args.hours)),
        "mode": args.mode,
        "checkpoint": ckpt_rel,
        "checkpoint_sha256": ckpt_sha,
        "source_tree_sha256": digests["source_tree_sha256"],
        "effective_config_sha256": digests["effective_config_sha256"],
        "source_bundle_sha256": digests["source_bundle_sha256"],
        "horizons": {
            str(hour): {
                key: float(np.mean([r["horizons"][str(hour)][key] for r in records]))
                for key in records[0]["horizons"][str(hour)]
            }
            for hour in sorted(set(args.hours))
        },
        "wall_seconds": time.monotonic() - t0,
        "records": records,
    }
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[POM3D EVAL | {args.protocol} | {args.split} | {args.mode}] "
        f"horizons={summary['hours']} hours",
        flush=True,
    )


if __name__ == "__main__":
    main()
