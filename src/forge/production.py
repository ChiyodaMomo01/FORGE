"""Resumable high-resolution reference trajectory production."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .cases import make_case
from .config import (
    ALL_CASES,
    COMPONENT_LABEL_CHANNELS,
    load_campaign,
    parent_tasks,
    save_json,
)
from .datasets.checks import physical_checks
from .datasets.storage import (
    complete,
    halt_path,
    initialize_manifest,
    paths,
    save_state,
)
from .diagnostics import diagnose
from .state import advance, to_device


def produce_parent(
    task,
    root,
    worker="direct",
    checkpoint_seconds=60.0,
    stop_after=None,
    device="cpu",
):
    """Produce a single parent trajectory with periodic restart checkpoints."""
    root = Path(root)
    meta, work = paths(root, task)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "restart.pt"
    progress = work / "progress.json"
    case = make_case(
        task["case"],
        {**task["reference_config"], "differentiable": False, "dtype": "float64"},
    )
    begun, cpu_begun = time.monotonic(), time.process_time()
    if checkpoint.exists():
        record = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if record["task"] != task:
            raise ValueError("restart configuration mismatch")
        record["state"] = to_device(record["state"], device, torch.float64)
    else:
        state = to_device(
            case.initial(task["seed"], task["hr_grid"], device=device),
            device,
            torch.float64,
        )
        record = {
            "task": task,
            "state": state,
            "burn_elapsed": 0.0,
            "frame": -1,
            "burn_target": task["burn_in"],
            "diagnostics": [],
            "burn_diagnostics": [],
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0,
            "owners": [],
        }
    record["owners"].append(
        {
            "worker": str(worker),
            "started": time.time(),
            "threads": torch.get_num_threads(),
        }
    )
    initial_wall, initial_cpu = record["wall_seconds"], record["cpu_seconds"]
    dt = task["sample_dt"]
    has_labels = task["split"] == "train" and task["case"] in COMPONENT_LABEL_CHANNELS
    if has_labels and (
        not callable(getattr(case, "closure_target", None))
        or case.action_channels != COMPONENT_LABEL_CHANNELS[task["case"]]
    ):
        raise ValueError("case does not implement its declared component-label contract")
    arrays = {}
    last_saved = time.monotonic()
    iterations = 0

    def persist(force=False):
        nonlocal last_saved
        if not force and time.monotonic() - last_saved < checkpoint_seconds:
            return
        for a in arrays.values():
            a.flush()
        record["wall_seconds"] = initial_wall + time.monotonic() - begun
        record["cpu_seconds"] = initial_cpu + time.process_time() - cpu_begun
        cpu_record = {
            **record,
            "state": to_device(record["state"], "cpu", torch.float64),
        }
        save_state(checkpoint, cpu_record)
        latest = (record["diagnostics"] or record["burn_diagnostics"] or [None])[-1]
        save_json(
            progress,
            {
                "status": "running",
                "case": task["case"],
                "split": task["split"],
                "parent": task["parent"],
                "worker": str(worker),
                "updated": time.time(),
                "frame": record["frame"],
                "frames": task["frames"],
                "burn_elapsed": record["burn_elapsed"],
                "burn_target": record["burn_target"],
                "wall_seconds": record["wall_seconds"],
                "cpu_seconds": record["cpu_seconds"],
                "latest_diagnostic": latest,
            },
        )
        last_saved = time.monotonic()

    def tick():
        nonlocal iterations
        iterations += 1
        tensors = (case.fields(record["state"]),)
        if not all(bool(x.isfinite().all()) for x in tensors):
            raise FloatingPointError(
                "nonfinite HR state at output boundary; last finite checkpoint retained"
            )
        persist()
        if stop_after is not None and iterations >= stop_after:
            persist(True)
            raise InterruptedError("requested checkpoint interruption")
        if halt_path(root, task).exists():
            persist(True)
            raise InterruptedError("case halted for review")

    def diagnostic(state):
        current = diagnose(case, state)
        history = record["diagnostics"] or record["burn_diagnostics"]
        rows = [history[0], history[-1], current] if history else [current]
        rows = list({float(row["time"]): row for row in rows}.values())
        checks = physical_checks(task["case"], rows)
        if checks["issues"]:
            raise FloatingPointError("; ".join(checks["issues"]))
        return current

    with torch.inference_mode():
        if not record["burn_diagnostics"]:
            record["burn_diagnostics"].append(diagnostic(record["state"]))
        persist(True)
        while record["burn_elapsed"] < record["burn_target"] - 1e-8:
            interval = min(dt, record["burn_target"] - record["burn_elapsed"])
            record["state"] = advance(case, record["state"], interval)
            record["burn_elapsed"] += interval
            cadence = dt * 10
            previous = float(record["burn_diagnostics"][-1]["time"])
            if (
                record["burn_elapsed"] - previous >= cadence - 1e-8
                or record["burn_elapsed"] >= record["burn_target"] - 1e-8
            ):
                record["burn_diagnostics"].append(diagnostic(record["state"]))
            tick()
        shapes = {
            "fields": (
                task["frames"] + 1,
                case.field_channels,
                task["lr_grid"],
                task["lr_grid"],
            )
        }
        if has_labels:
            shapes["actions"] = (
                task["frames"],
                case.action_channels,
                task["lr_grid"],
                task["lr_grid"],
            )
        for key, shape in shapes.items():
            path = work / (key + ".npy")
            mode = "r+" if path.exists() else "w+"
            if record["frame"] >= 0 and mode == "w+":
                raise FileNotFoundError("checkpoint array missing: " + str(path))
            arrays[key] = np.lib.format.open_memmap(path, mode=mode, dtype=np.float32, shape=shape)
        if record["frame"] < 0:
            coarse = case.restrict(record["state"], task["lr_grid"])
            save_state(work / "initial.pt", to_device(coarse, "cpu", torch.float64))
            save_state(
                work / "hr-record-start.pt",
                to_device(record["state"], "cpu", torch.float64),
            )
            arrays["fields"][0] = case.fields(coarse)[0].detach().cpu().float().numpy()
            record["diagnostics"].append(diagnostic(record["state"]))
            record["frame"] = 0
            persist(True)
        while record["frame"] < task["frames"]:
            i = record["frame"]
            if has_labels:
                arrays["actions"][i] = (
                    case.closure_target(record["state"], task["lr_grid"])[0]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                if not np.isfinite(arrays["actions"][i]).all():
                    raise FloatingPointError("nonfinite closure label")
            record["state"] = advance(case, record["state"], dt)
            diag = diagnostic(record["state"])
            arrays["fields"][i + 1] = (
                case.fields(case.restrict(record["state"], task["lr_grid"]))[0]
                .detach()
                .cpu()
                .float()
                .numpy()
            )
            if not np.isfinite(arrays["fields"][i + 1]).all():
                raise FloatingPointError("nonfinite stored fields")
            record["diagnostics"].append(diag)
            record["frame"] += 1
            tick()
        persist(True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    path = meta.with_suffix(".npz")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as out:
        np.savez_compressed(out, times=np.arange(task["frames"] + 1) * dt, **arrays)
    temporary.replace(path)
    save_state(
        meta.with_name(meta.stem + "-initial.pt"),
        torch.load(work / "initial.pt", weights_only=False),
    )
    f = arrays["fields"]
    minimum = f.min(axis=(0, 2, 3))
    mean_drift = abs(
        f.mean(axis=(-2, -1), dtype=np.float64) - f[0].mean(axis=(-2, -1), dtype=np.float64)
    ).max()
    summary = {
        **task,
        "status": "complete",
        "reference_qualification": "high-resolution float64 physical and integrity verification",
        "execution_device": str(case.fields(record["state"]).device),
        "evolution_dtype": str(case.fields(record["state"]).dtype),
        "solver": case.metadata(),
        "field_shape": list(f.shape),
        "component_labels": has_labels,
        "physical_axes": ["y", "x"],
        "actual_burn_in": record["burn_elapsed"],
        "diagnostic_precision": "HR float64 before restriction",
        "diagnostics": record["diagnostics"],
        "burn_diagnostics": record["burn_diagnostics"],
        "owners": record["owners"],
        "wall_seconds": initial_wall + time.monotonic() - begun,
        "cpu_seconds": initial_cpu + time.process_time() - cpu_begun,
        "minimum_field_value": float(minimum.min()),
        "channel_minima": minimum.tolist(),
        "max_channel_mean_drift": float(mean_drift),
        "bytes": path.stat().st_size,
        "initial_restart": "case-defined restriction of HR recording start",
    }
    save_json(meta, summary)
    save_json(
        progress,
        {
            "status": "complete",
            "updated": time.time(),
            "frame": task["frames"],
            "worker": str(worker),
        },
    )
    for key in arrays:
        (work / (key + ".npy")).unlink()
    return summary


def produce_group(tasks, root, device="cpu", rank=0):
    if str(device) != "cpu":
        raise ValueError("reference production is CPU-only")
    return [produce_parent(t, root, str(rank)) for t in tasks]


def ordered_tasks(campaign, cases):
    tasks = [t for t in parent_tasks(campaign) if t["case"] in cases]
    tasks.sort(
        key=lambda t: (
            t["seed"]
            - campaign["cases"][t["case"]]["seed_base"]
            - sum(
                n
                for s, n in list(campaign["splits"].items())[
                    : list(campaign["splits"]).index(t["split"])
                ]
            ),
            list(ALL_CASES).index(t["case"]),
            list(campaign["splits"]).index(t["split"]),
        )
    )
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--root")
    p.add_argument("--worker", default="cpu-0")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cases", nargs="+", choices=ALL_CASES, default=None)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    campaign = load_campaign(a.config)
    selected_cases = list(campaign["cases"].keys()) if a.cases is None else a.cases
    if a.plan:
        print(
            json.dumps(
                {"campaign": campaign, "parent_count": len(parent_tasks(campaign))},
                indent=2,
            )
        )
        return
    if not a.root or a.threads < 1:
        p.error("--root and positive --threads required")
    if not str(a.device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    root = Path(a.root)
    initialize_manifest(root, campaign)
    done = 0
    for task in ordered_tasks(campaign, selected_cases):
        meta, work = paths(root, task)
        if halt_path(root, task).exists() or complete(root, task):
            continue
        try:
            result = produce_parent(task, root, a.worker, device=a.device)
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "case": task["case"],
                        "parent": task["parent"],
                        "seconds": result["wall_seconds"],
                    }
                ),
                flush=True,
            )
            done += 1
        except Exception:
            failure = {
                "task": task,
                "worker": str(a.worker),
                "time": time.time(),
                "traceback": traceback.format_exc(),
            }
            save_json(work / "failure.json", failure)
            halt = halt_path(root, task)
            if not halt.exists():
                save_json(halt, failure)
            print(
                json.dumps(
                    {
                        "status": "halted",
                        "case": task["case"],
                        "parent": task["parent"],
                    }
                ),
                flush=True,
            )
        if a.limit is not None and done >= a.limit:
            break


if __name__ == "__main__":
    main()
