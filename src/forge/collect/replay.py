"""Replay frozen HR recording starts, tap ACTUAL G calls, retain raw I/O.

Only this trusted collector imports physical cases. It never uses a
closure_target, AD optimized action, trained network, or formula-derived label.
Published original data are read-only and new data have a separate root.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from ..cases import make_case
from ..core.io import file_digest, save
from ..schemas import (
    canonical_actual_g_schema,
    canonical_collection_protocol,
    is_turbulence,
)
from ..state import advance, to_device
from .restrict import restrict

TURBULENCE_PAIR_STRIDE = 4


def usage():
    row = resource.getrusage(resource.RUSAGE_SELF)
    return row.ru_utime + row.ru_stime


def replay(task):
    torch.set_num_threads(1)
    if len(task) == 5:
        source, destination, metadata_path, source_revision, worker_device = task
    else:
        source, destination, metadata_path, source_revision = task
        worker_device = "cpu"
    if not str(worker_device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        worker_device = "cpu"
    source, destination, metadata_path = map(Path, (source, destination, metadata_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    case_name, split, parent = metadata["case"], metadata["split"], metadata["parent"]
    turbulence = is_turbulence(case_name)
    out = destination / case_name / split / parent
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return json.loads((out / "complete.json").read_text(encoding="utf-8"))
    if any(out.glob("*.npy")):
        raise FileExistsError("partial replay retained; use a new attempt root")
    begun, before = time.monotonic(), usage()
    state_path = source / ".work" / case_name / split / parent / "hr-record-start.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state = to_device(state, worker_device, torch.float64)
    case = make_case(case_name, {**metadata["reference_config"], "dtype": "float64"})
    frames, dt = int(metadata["frames"]), float(metadata["sample_dt"])
    hr, lr = int(metadata["hr_grid"]), int(metadata["lr_grid"])
    pairs = split in ("train", "validation")
    spec = {
        "cahn_hilliard": (
            1,
            2,
            1,
            "c,kappa*Lap(c)",
            "full cubic chemical-potential component",
            "conservative block average",
        ),
        "gray_scott": (
            2,
            4,
            1,
            "u,v,D_u*Lap(u)/.05,D_v*Lap(v)/.05",
            "full autocatalytic reaction rate u*v^2",
            "conservative block average",
        ),
        "kolmogorov_ns": (
            1,
            6,
            2,
            "omega,psi,u,v,omega_x,omega_y",
            "full 3/2-dealiased advective vorticity flux (u*omega,v*omega)",
            "Fourier Galerkin truncation with zeroed coarse Nyquist lines",
        ),
        "mhd2d": (
            2,
            8,
            3,
            "omega,a,psi,j,u,v,bx,by",
            "full 3/2-dealiased Reynolds-Maxwell flux and out-of-plane EMF",
            "Fourier Galerkin truncation with zeroed coarse Nyquist lines",
        ),
    }
    channels, inputs, outputs, input_contract, output_contract, restriction_name = spec[case_name]
    pair_stride = TURBULENCE_PAIR_STRIDE if turbulence else 1
    pair_intervals = list(range(0, frames, pair_stride)) if pairs else []
    pair_interval_to_slot = {k: idx for idx, k in enumerate(pair_intervals)}
    num_pairs = len(pair_intervals)

    if turbulence:
        ref_audit_frames = sorted(set([0, frames // 2, frames]))
        ref_audit_map = {k: idx for idx, k in enumerate(ref_audit_frames)}
        pair_audit_slots = sorted(set([0, num_pairs // 2, num_pairs - 1])) if pairs else []
        pair_audit_map = {slot: idx for idx, slot in enumerate(pair_audit_slots)}
        shapes = {
            "reference_hr_audit": (len(ref_audit_frames), channels, hr, hr),
            "reference_lr": (frames + 1, channels, lr, lr),
        }
        if pairs:
            shapes.update(
                input_hr_audit=(len(pair_audit_slots), inputs, hr, hr),
                output_hr_audit=(len(pair_audit_slots), outputs, hr, hr),
                input_lr=(num_pairs, inputs, lr, lr),
                output_lr=(num_pairs, outputs, lr, lr),
            )
    else:
        hr_audit_pairs = (
            set(range(frames)) if (hr < 768 or frames <= 16) else {0, frames // 2, frames - 1}
        )
        hr_audit_refs = (
            set(range(frames + 1)) if (hr < 768 or frames <= 16) else {0, frames // 2, frames}
        )
        shapes = {
            "reference_hr": (frames + 1, channels, hr, hr),
            "reference_lr": (frames + 1, channels, lr, lr),
        }
        if pairs:
            shapes.update(
                input_hr=(frames, inputs, hr, hr),
                output_hr=(frames, outputs, hr, hr),
                input_lr=(frames, inputs, lr, lr),
                output_lr=(frames, outputs, lr, lr),
            )

    arrays = {}
    for key, shape in shapes.items():
        arrays[key] = np.lib.format.open_memmap(
            out / (key + ".npy"), mode="w+", dtype=np.float64, shape=shape
        )
    with np.load(metadata_path.with_suffix(".npz"), allow_pickle=False) as original:
        old_fields, old_times = original["fields"], original["times"]
    if old_fields.shape != shapes["reference_lr"] or not np.allclose(
        np.diff(old_times), dt, rtol=0, atol=1e-8
    ):
        raise ValueError("original field/time contract mismatch")
    np.save(out / "times.npy", old_times)
    pending_slot = None
    calls, saved_calls, sampled_indices, interval_calls = 0, 0, [], []
    actual_g = case.g

    def tap(public_input):
        nonlocal pending_slot, calls, saved_calls
        public_output = actual_g(public_input)
        if pending_slot is not None:
            slot = pending_slot
            for key, value in (("input", public_input), ("output", public_output)):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("nonfinite actual public G I/O")
                val_cpu = value.detach().cpu()
                if turbulence:
                    if slot in pair_audit_map:
                        arrays[key + "_hr_audit"][pair_audit_map[slot]] = val_cpu[0].numpy()
                else:
                    if slot in hr_audit_pairs:
                        arrays[key + "_hr"][slot] = val_cpu[0].numpy()
                arrays[key + "_lr"][slot] = restrict(val_cpu, lr, case_name)[0].numpy()
            sampled_indices.append(calls)
            saved_calls += 1
            pending_slot = None
        calls += 1
        return public_output

    case.g = tap
    exact = True
    max_abs = 0.0
    try:
        with torch.inference_mode():
            for k in range(frames + 1):
                field = case.fields(state)
                coarse = case.fields(case.restrict(state, lr))
                if field.dtype != torch.float64 or not bool(torch.isfinite(field).all()):
                    raise ValueError("finite float64 HR state required")
                field_cpu = field.detach().cpu()
                coarse_cpu = coarse.detach().cpu()
                if turbulence:
                    if k in ref_audit_map:
                        arrays["reference_hr_audit"][ref_audit_map[k]] = field_cpu[0].numpy()
                else:
                    if k in hr_audit_refs:
                        arrays["reference_hr"][k] = field_cpu[0].numpy()
                arrays["reference_lr"][k] = coarse_cpu[0].numpy()
                rounded = coarse_cpu[0].float().numpy()
                exact &= bool(np.array_equal(rounded, old_fields[k]))
                max_abs = max(max_abs, float(np.max(np.abs(rounded - old_fields[k]))))
                if not exact:
                    raise ValueError(
                        f"frozen replay differs from published LR field at {k}: {max_abs}"
                    )
                if k == frames:
                    break
                pending_slot = pair_interval_to_slot.get(k) if pairs else None
                calls_before = calls
                state = advance(case, state, dt)
                interval_calls.append(calls - calls_before)
                if pending_slot is not None:
                    raise ValueError("no actual G invocation in recording interval")
                if k % 10 == 0:
                    save(
                        out / "progress.json",
                        {
                            "status": "replaying",
                            "frame": k + 1,
                            "frames": frames,
                            "actual_g_calls": calls,
                            "saved_pairs": saved_calls,
                            "wall_seconds": time.monotonic() - begun,
                            "cpu_seconds": usage() - before,
                            "device": str(worker_device),
                            "source_revision": source_revision,
                        },
                    )
        for value in arrays.values():
            value.flush()
        np.save(out / "pair_call_index.npy", np.asarray(sampled_indices, np.int64))
        np.save(out / "interval_g_calls.npy", np.asarray(interval_calls, np.int64))
        if turbulence:
            np.save(
                out / "pair_interval_indices.npy",
                np.asarray(pair_intervals, np.int64),
            )
        if pairs and saved_calls != num_pairs:
            raise ValueError("incomplete actual G pair coverage")
        record = {
            "schema": canonical_actual_g_schema(case_name),
            "status": "complete",
            "case": case_name,
            "split": split,
            "parent": parent,
            "seed": metadata["seed"],
            "frames": frames,
            "sample_dt": dt,
            "hr_grid": hr,
            "lr_grid": lr,
            "input_channels": inputs,
            "output_channels": outputs,
            "field_channels": channels,
            "source_start_sha256": file_digest(state_path),
            "source_metadata_sha256": file_digest(metadata_path),
            "source_revision": source_revision,
            "pairs": saved_calls,
            "pair_stride": pair_stride,
            "actual_g_calls": calls,
            "pair_sampling": (
                f"first ACTUAL G call at saved intervals 0..{frames - 1} step {pair_stride}; stage0; no added G evaluation"
                if turbulence
                else "first ACTUAL G call at the beginning of each saved interval; stage0; no added G evaluation"
            ),
            "input_contract": input_contract,
            "output_contract": output_contract,
            "label_source": "return value of the SAME G invocation consumed by the HR integrator",
            "restriction": restriction_name,
            "original_lr_bitwise_equal": exact,
            "maximum_original_lr_difference": max_abs,
            **(
                {
                    "hr_audit_ref_frames": ref_audit_frames,
                    "hr_audit_pair_slots": pair_audit_slots,
                }
                if turbulence
                else {}
            ),
            "arrays": {
                key: {"shape": list(a.shape), "dtype": str(a.dtype)} for key, a in arrays.items()
            },
            "execution_device": str(worker_device),
            "cpu_seconds": usage() - before,
            "wall_seconds": time.monotonic() - begun,
            "bytes": sum(p.stat().st_size for p in out.glob("*.npy")),
        }
        save(out / "complete.json", record)
        return record
    except BaseException as error:
        for value in arrays.values():
            value.flush()
        save(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "cpu_seconds": usage() - before,
                "wall_seconds": time.monotonic() - begun,
                "saved_pairs": saved_calls,
                "actual_g_calls": calls,
            },
        )
        raise


def collect_case(
    source: str | Path,
    output: str | Path,
    case_name: str,
    *,
    workers: int = 8,
    devices: str = "",
    source_revision: str = "forge-release",
    expected_splits: dict[str, int] | None = None,
) -> dict:
    root, out = Path(source), Path(output)
    turbulence = is_turbulence(case_name)
    canonical_splits = (
        {"train": 24, "validation": 8, "test_id": 8}
        if turbulence
        else {"train": 48, "validation": 16, "test_id": 16}
    )
    official_split = expected_splits is None or dict(expected_splits) == canonical_splits
    split_targets = canonical_splits if official_split else dict(expected_splits)
    if not official_split:
        for sp in ("train", "validation", "test_id"):
            cnt = int(split_targets.get(sp, 0))
            if cnt <= 0 or cnt % 4 != 0:
                raise ValueError(f"custom split {sp!r}={cnt} must be a positive multiple of 4")

    gpu_list = [f"cuda:{tok.strip()}" for tok in devices.split(",") if tok.strip()]
    tasks, identities = [], set()
    task_idx = 0
    for split in ("train", "validation", "test_id"):
        paths = sorted((root / case_name / split).glob("*.json"))
        expected = int(split_targets[split])
        complete = [
            p
            for p in paths
            if json.loads(p.read_text(encoding="utf-8")).get("status") == "complete"
        ]
        if len(complete) != expected:
            raise ValueError(
                f"original split parent counts differ for {case_name}/{split}: "
                f"got {len(complete)}, expected {expected}"
            )
        for path in complete:
            identity = json.loads(path.read_text(encoding="utf-8"))["seed"]
            if identity in identities:
                raise ValueError("original parent leakage")
            identities.add(identity)
            worker_dev = gpu_list[task_idx % len(gpu_list)] if gpu_list else "cpu"
            task_idx += 1
            tasks.append((str(root), str(out), str(path), source_revision, worker_dev))
    record = {
        "status": "running",
        "case": case_name,
        "protocol": canonical_collection_protocol(case_name),
        "official_benchmark_split": official_split,
        **({} if official_split else {"custom_split_counts": split_targets}),
        "parents": len(tasks),
        "workers": workers,
        "devices": gpu_list or ["cpu"],
        "completed": [],
        "failures": [],
        "started_unix": time.time(),
        "source_revision": source_revision,
    }
    save(out / case_name / "collection.json", record)
    if workers <= 1:
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            for task in tasks:
                try:
                    record["completed"].append(replay(task))
                except BaseException as error:
                    record["failures"].append(
                        {
                            "metadata": Path(task[2]).name,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                save(out / case_name / "collection.json", record)
        finally:
            if prev_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(replay, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    record["completed"].append(future.result())
                except BaseException as error:
                    record["failures"].append(
                        {
                            "metadata": Path(futures[future][2]).name,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                save(out / case_name / "collection.json", record)
    record["status"] = "failed" if record["failures"] else "complete"
    record["finished_unix"] = time.time()
    save(out / case_name / "collection.json", record)
    if record["failures"]:
        raise RuntimeError("HR replay failures retained; training is not permitted")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=("cahn_hilliard", "gray_scott", "kolmogorov_ns", "mhd2d"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--devices",
        default="",
        help="Optional comma-separated visible CUDA indices (e.g. 0,1,2,3) for GPU float64 replay",
    )
    parser.add_argument("--source-revision", default="forge-release")
    args = parser.parse_args(argv)
    collect_case(
        args.source,
        args.output,
        args.case,
        workers=args.workers,
        devices=args.devices,
        source_revision=args.source_revision,
    )


if __name__ == "__main__":
    main()
