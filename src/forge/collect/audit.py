"""Independent stored-pair restriction and temporal/provenance audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..core.io import file_digest, save
from ..paired.data import audit as metadata_audit
from ..paired.data import resolve_pair_record_path
from ..schemas import canonical_collection_protocol, is_turbulence
from .restrict import restrict


def _independent_fourier_restrict_np(arr: np.ndarray, n: int) -> np.ndarray:
    """Analytical complex 2D DFT Fourier truncation independent of spectral_restrict."""
    old = arr.shape[-1]
    if arr.shape[-2] != old or n < 2 or old % n or n % 2 or old % 2:
        raise ValueError("even integer square-grid restriction required")
    fh = np.fft.fft2(np.asarray(arr, dtype=np.float64), axes=(-2, -1))
    ch = np.zeros((*arr.shape[:-2], n, n), dtype=np.complex128)
    half = n // 2
    modes = [m for m in range(-half + 1, half)]
    fine_idx = [m % old for m in modes]
    coarse_idx = [m % n for m in modes]
    ch[..., np.ix_(coarse_idx, coarse_idx)[0], np.ix_(coarse_idx, coarse_idx)[1]] = fh[
        ..., np.ix_(fine_idx, fine_idx)[0], np.ix_(fine_idx, fine_idx)[1]
    ]
    ch *= (n / old) ** 2
    return np.fft.ifft2(ch, axes=(-2, -1)).real


def audit(root, case_name, original_root=None):
    torch.set_num_threads(1)
    case_root = Path(root) / case_name
    info = metadata_audit(root, case_name)
    turbulence = is_turbulence(case_name)
    if turbulence and original_root is None:
        raise ValueError(
            "turbulence audit requires original_root to verify substeps and F commutation"
        )
    checks = []
    for split, group in info["splits"].items():
        for raw_path in group["records"]:
            path = resolve_pair_record_path(case_root, raw_path)
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("original_lr_bitwise_equal") is not True:
                raise ValueError("original LR replay is not bitwise equal")
            folder = path.parent
            times = np.load(folder / "times.npy")
            expected_times = np.arange(meta["frames"] + 1) * meta["sample_dt"]
            if not np.isfinite(times).all() or not np.allclose(
                times, expected_times, rtol=0, atol=1e-12
            ):
                raise ValueError("stored timeline mismatch")
            calls = np.load(folder / "interval_g_calls.npy")
            if calls.shape != (meta["frames"],) or np.any(calls < 1):
                raise ValueError("invalid actual call coverage")

            source = None
            if original_root is not None:
                source = json.loads(
                    (
                        Path(original_root) / case_name / split / (meta["parent"] + ".json")
                    ).read_text(encoding="utf-8")
                )
            if turbulence:
                ref_substeps = int(source["reference_config"].get("substeps", 1))
                expected_calls_per_interval = 4 * ref_substeps
                if not np.all(calls == expected_calls_per_interval):
                    raise ValueError(
                        f"expected {expected_calls_per_interval} actual G calls per interval, got {calls.tolist()[:4]}"
                    )

            sampled = np.load(folder / "pair_call_index.npy")
            cum_calls = np.cumsum(np.r_[0, calls[:-1]])
            if turbulence:
                pair_stride = int(meta.get("pair_stride", 4))
                pair_intervals = np.load(folder / "pair_interval_indices.npy")
                expected_intervals = (
                    np.arange(0, meta["frames"], pair_stride, dtype=np.int64)
                    if meta["pairs"]
                    else np.zeros((0,), dtype=np.int64)
                )
                if not np.array_equal(pair_intervals, expected_intervals):
                    raise ValueError("turbulence pair interval indices mismatch")
                if meta["pairs"] and not np.array_equal(sampled, cum_calls[pair_intervals]):
                    raise ValueError("pair is not first actual G call of its sampled interval")
            else:
                if meta["pairs"] and not np.array_equal(sampled, cum_calls):
                    raise ValueError("pair is not first actual G call of its saved interval")

            errors = {}
            indep_pair_errors = {}
            if turbulence:
                pair_audit_slots = list(meta.get("hr_audit_pair_slots", []))
                ref_audit_frames = list(meta.get("hr_audit_ref_frames", []))
                expected_ref_frames = sorted(set([0, meta["frames"] // 2, meta["frames"]]))
                if ref_audit_frames != expected_ref_frames:
                    raise ValueError(
                        f"turbulence hr_audit_ref_frames mismatch: {ref_audit_frames} != {expected_ref_frames}"
                    )
                if meta["pairs"]:
                    num_pairs = int(len(expected_intervals))
                    expected_pair_slots = sorted(set([0, num_pairs // 2, num_pairs - 1]))
                    if not pair_audit_slots or pair_audit_slots != expected_pair_slots:
                        raise ValueError(
                            f"turbulence hr_audit_pair_slots mismatch: {pair_audit_slots} != {expected_pair_slots}"
                        )
                    for key in ("input", "output"):
                        hr = np.load(folder / (key + "_hr_audit.npy"), mmap_mode="r")
                        lr = np.load(folder / (key + "_lr.npy"), mmap_mode="r")
                        if hr.shape[0] != len(expected_pair_slots):
                            raise ValueError(f"unexpected {key}_hr_audit slot count")
                        max_indep = 0.0
                        for idx_in_audit, slot in enumerate(pair_audit_slots):
                            hr_slice = np.array(hr[idx_in_audit : idx_in_audit + 1])
                            lr_slice = np.asarray(lr[slot])
                            if not np.isfinite(hr_slice).all() or not np.isfinite(lr_slice).all():
                                raise ValueError("nonfinite stored HR/LR pair array")
                            expected = restrict(
                                torch.from_numpy(hr_slice),
                                meta["lr_grid"],
                                case_name,
                            )[0].numpy()
                            if not np.array_equal(expected, lr_slice):
                                raise ValueError("actual stored pair restriction mismatch")
                            indep = _independent_fourier_restrict_np(hr_slice, meta["lr_grid"])[0]
                            diff_indep = float(np.max(np.abs(indep - lr_slice)))
                            scale_lr = max(1.0, float(np.max(np.abs(lr_slice))))
                            if not np.isfinite(diff_indep) or diff_indep > 1e-12 * scale_lr:
                                raise ValueError(
                                    f"independent analytical Fourier restriction mismatch on {key}: {diff_indep}"
                                )
                            max_indep = max(max_indep, diff_indep)
                        errors[key] = 0.0
                        indep_pair_errors[key] = max_indep
                else:
                    if pair_audit_slots:
                        raise ValueError("held-out parent must have empty hr_audit_pair_slots")

                hr_ref = np.load(folder / "reference_hr_audit.npy", mmap_mode="r")
                lr = np.load(folder / "reference_lr.npy", mmap_mode="r")
                if hr_ref.shape[0] != len(expected_ref_frames):
                    raise ValueError("unexpected reference_hr_audit frame count")
                maximum = 0.0
                maximum_indep_ref = 0.0
                for idx_in_audit, k in enumerate(ref_audit_frames):
                    hr_slice = np.array(hr_ref[idx_in_audit : idx_in_audit + 1])
                    lr_frame = np.asarray(lr[k])
                    if not np.isfinite(hr_slice).all() or not np.isfinite(lr_frame).all():
                        raise ValueError("nonfinite reference HR/LR array")
                    expected = restrict(
                        torch.from_numpy(hr_slice),
                        meta["lr_grid"],
                        case_name,
                    )[0].numpy()
                    difference = float(np.max(np.abs(expected - lr_frame)))
                    maximum = max(maximum, difference)
                    scale_ref = max(1.0, float(np.max(np.abs(lr_frame))))
                    if difference > 1e-12 * scale_ref:
                        raise ValueError("HR/reference restriction inconsistency")
                    indep_ref = _independent_fourier_restrict_np(hr_slice, meta["lr_grid"])[0]
                    diff_indep_ref = float(np.max(np.abs(indep_ref - lr_frame)))
                    maximum_indep_ref = max(maximum_indep_ref, diff_indep_ref)
                    if not np.isfinite(diff_indep_ref) or diff_indep_ref > 1e-12 * scale_ref:
                        raise ValueError(
                            f"independent analytical reference restriction inconsistency: {diff_indep_ref}"
                        )
                paired_check_indices = (
                    [int(pair_intervals[s]) for s in pair_audit_slots] if meta["pairs"] else []
                )
            else:
                indices = sorted(set([0, meta["frames"] // 2, meta["frames"] - 1]))
                if meta["pairs"]:
                    for key in ("input", "output"):
                        hr = np.load(folder / (key + "_hr.npy"), mmap_mode="r")
                        lr = np.load(folder / (key + "_lr.npy"), mmap_mode="r")
                        for k in indices:
                            expected = restrict(
                                torch.from_numpy(np.array(hr[k : k + 1])),
                                meta["lr_grid"],
                                case_name,
                            )[0].numpy()
                            if not np.array_equal(expected, lr[k]):
                                raise ValueError("actual stored pair restriction mismatch")
                        errors[key] = 0.0
                hr = np.load(folder / "reference_hr.npy", mmap_mode="r")
                lr = np.load(folder / "reference_lr.npy", mmap_mode="r")
                maximum = 0.0
                for k in sorted(set([0, meta["frames"] // 2, meta["frames"]])):
                    expected = restrict(
                        torch.from_numpy(np.array(hr[k : k + 1])),
                        meta["lr_grid"],
                        case_name,
                    )[0].numpy()
                    difference = float(np.max(np.abs(expected - lr[k])))
                    maximum = max(maximum, difference)
                    if difference > 1e-12 * max(1.0, float(np.max(np.abs(lr[k])))):
                        raise ValueError("HR/reference restriction inconsistency")
                paired_check_indices = indices if meta["pairs"] else []

            row_entry = {
                "split": split,
                "parent": meta["parent"],
                "paired_sample_indices": paired_check_indices,
                "pair_restriction_errors": errors,
                "reference_restriction_max_abs": maximum,
                "all_original_lr_frames_bitwise_equal": meta["original_lr_bitwise_equal"],
                "complete_metadata_sha256": file_digest(path),
            }
            if turbulence:
                row_entry["independent_dft_pair_max_abs"] = indep_pair_errors
                row_entry["independent_dft_reference_max_abs"] = maximum_indep_ref
            checks.append(row_entry)

            if meta["pairs"] and source is not None:
                from ..cases import make_case

                native = make_case(case_name, source["deployment_config"])
                x = np.load(folder / "input_lr.npy", mmap_mode="r")
                diffs = []
                rms = []
                slot_and_frame = (
                    [(s, int(pair_intervals[s])) for s in pair_audit_slots]
                    if turbulence
                    else [(k, k) for k in indices]
                )
                if not slot_and_frame:
                    raise ValueError("missing F commutation audit slots")
                for slot, frame_k in slot_and_frame:
                    state = native.state_from_fields(
                        torch.from_numpy(np.array(lr[frame_k : frame_k + 1]))
                    )
                    public = native.observe(state)
                    expected = np.asarray(x[slot : slot + 1])
                    actual = public.numpy()
                    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                        raise ValueError("nonfinite public F values in audit")
                    diffs.append(np.square(actual - expected).mean((0, 2, 3)))
                    rms.append(np.square(expected).mean((0, 2, 3)))
                error = np.sqrt(np.mean(diffs, axis=0))
                scale = np.maximum(np.sqrt(np.mean(rms, axis=0)), 1e-30)
                rel_error = error / scale
                if not np.isfinite(error).all() or not np.isfinite(rel_error).all():
                    raise ValueError("nonfinite F commutation error")
                if turbulence and float(np.max(rel_error)) > 1e-12:
                    raise ValueError(
                        f"turbulence F commutation relative RMSE exceeded 1e-12: {rel_error.tolist()}"
                    )
                checks[-1]["deployment_F_vs_restricted_HR_F_rmse"] = error.tolist()
                checks[-1]["deployment_F_vs_restricted_HR_F_relative_rmse"] = rel_error.tolist()
                checks[-1]["F_comparison_qualification"] = (
                    "exact spectral commutation gate (enforced <= 1e-12)"
                    if turbulence
                    else "input distribution shift diagnostic, not assumed exact commutation"
                )
    result = {
        "status": "passed",
        "case": case_name,
        "protocol": canonical_collection_protocol(case_name),
        "parents": len(checks),
        "checks": checks,
        "scope": "all parent/time/call identities; every-frame replay against original; stored HR/LR arrays independently checked at start/middle/end via both rfft2 and analytical complex DFT; spectral turbulence F commutation enforced <= 1e-12",
        "effect_claim": "none; this is data-contract verification",
    }
    save(case_root / "restriction-audit.json", result)
    return info
