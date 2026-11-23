"""Checkpoint, budget-milestone, cost-fraction, and out-of-place resume state helpers for Study."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..core.io import file_digest
from .client import merge_scores
from .common import array, checkpoint, digest, load_model, save, set_model
from .events import anchors


def _relative_or_name(target: Path, anchor: Path) -> str:
    try:
        return target.resolve().relative_to(anchor.resolve().parent).as_posix()
    except ValueError:
        return target.name


def load_merged_oracle_query(
    shard_dirs: list[Path],
    has_gpu_shards: bool,
    prev: Path,
    label: str,
    expected_trials: list[int] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    for odir in shard_dirs:
        found = None
        ledger_path = odir / "ledger.jsonl"
        if ledger_path.is_file():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("label") == label:
                    found = obj
                    break
        if found is None:
            for qpub in sorted(odir.glob("query-*/public.json")):
                obj = json.loads(qpub.read_text(encoding="utf-8"))
                if obj.get("label") == label:
                    found = obj
                    break
        if found is not None:
            rows.append(found)
    if not rows:
        raise RuntimeError(f"missing saved query {label!r} in {prev.name}")
    if not has_gpu_shards and len(rows) == 1:
        return rows[0]
    sel = (
        list(expected_trials)
        if expected_trials is not None
        else sorted({t for r in rows for t in r["trials"]})
    )
    return merge_scores(rows, sel, kwargs or {})


def resume_study_from_attempt(study, prev_folder: str | Path) -> tuple[int, int]:
    prev = Path(prev_folder).resolve()
    if not prev.is_dir():
        raise FileNotFoundError(f"missing resume attempt directory: {prev}")
    if prev == study.folder.resolve():
        raise ValueError("never resume in-place; new output root required")
    if getattr(study.recipe, "actor_fp64", False):
        raise ValueError("resume_from_attempt only supports fp32 main")

    prev_init_path = prev / "initialization.json"
    prev_init = json.loads(prev_init_path.read_text(encoding="utf-8"))
    if prev_init.get("policy_hash") != digest(study.initial):
        raise ValueError("resume attempt initial policy_hash mismatch")

    gpu_oracles = sorted(
        prev.glob("gpu-*/oracle"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    shard_dirs = (
        gpu_oracles if gpu_oracles else ([prev / "oracle"] if (prev / "oracle").is_dir() else [])
    )
    if not shard_dirs:
        raise FileNotFoundError(f"no oracle ledger directories found in {prev.name}")

    if study.reference:
        retained = json.loads((study.reference / "common/result.json").read_text(encoding="utf-8"))
        if retained["policy_hash"] != digest(study.initial):
            raise ValueError("retained common policy mismatch")
        study.train = retained["training"]
        val0 = retained["evaluations"]["validation"]
    else:
        study.train = load_merged_oracle_query(
            shard_dirs, bool(gpu_oracles), prev, "shared-pretrained-train"
        )
        val0 = load_merged_oracle_query(
            shard_dirs, bool(gpu_oracles), prev, "shared-pretrained-validation"
        )

    study.best = (float(val0["mean_mse"]), 0, study.center.copy())
    study.best_score = val0
    study.initial_validation_mse = float(val0["mean_mse"])
    study.significant_best = float(val0["mean_mse"])
    study.bad_checks = 0
    study.stopped = False

    anchor_trials = study.batch()
    logged = load_merged_oracle_query(
        shard_dirs,
        bool(gpu_oracles),
        prev,
        "pretrained-public-anchors",
        expected_trials=anchor_trials,
        kwargs={"capacity": 1, "label": "pretrained-public-anchors", "seed": 2000},
    )
    if (
        logged.get("policy_hash") != digest(study.initial)
        or int(logged.get("seed", -1)) != 2000
        or [int(t) for t in logged.get("trials", [])] != [int(t) for t in anchor_trials]
    ):
        raise ValueError("saved pretrained-public-anchors provenance mismatch")
    study.obs = anchors(logged, 16)

    round_files = sorted((prev / "rounds").glob("round-*.json"))
    if not round_files:
        raise ValueError(f"no completed round-*.json files found in {prev.name}")
    k_done = len(round_files)
    (study.folder / "rounds").mkdir(parents=True, exist_ok=True)
    best_round_expected = 0
    best_mse_expected = float(val0["mean_mse"])
    total_probe_retries = 0

    for idx, rf in enumerate(round_files, start=1):
        rd = json.loads(rf.read_text(encoding="utf-8"))
        if int(rd.get("round", -1)) != idx:
            raise ValueError(f"non-contiguous round index: expected {idx}, got {rd.get('round')}")
        expected_trials = study.batch()
        actual_trials = [int(t) for t in rd["center_score"]["trials"]]
        if actual_trials != [int(t) for t in expected_trials]:
            raise ValueError(
                f"round {idx} batch trial replay mismatch: {actual_trials} != {expected_trials}"
            )
        total_probe_retries = int(rd.get("total_probe_retries", total_probe_retries))
        if "validation" in rd and rd["validation"] is not None:
            val = rd["validation"]
            if not val["failures"] and val["mean_mse"] < best_mse_expected:
                best_mse_expected = float(val["mean_mse"])
                best_round_expected = idx
                study.best_score = val
            if not val["failures"] and val["mean_mse"] < study.significant_best * (
                1 - study.recipe.plateau_relative
            ):
                study.significant_best = float(val["mean_mse"])
                study.bad_checks = 0
            else:
                study.bad_checks += 1
            study.stopped = study.bad_checks >= study.recipe.patience
            es = rd.get("early_stopping", {})
            if (
                not np.isclose(
                    float(es.get("significant_best", -1.0)),
                    float(study.significant_best),
                    rtol=0.0,
                    atol=1e-15,
                )
                or int(es.get("bad_checks", -1)) != int(study.bad_checks)
                or bool(es.get("stop", False)) != bool(study.stopped)
            ):
                raise ValueError(f"round {idx} early_stopping replay mismatch")
        study.history.append(rd)
        shutil.copy2(rf, study.folder / "rounds" / rf.name)
        prev_rdir = prev / f"round-{idx}"
        if prev_rdir.is_dir():
            dst_rdir = study.folder / f"round-{idx}"
            if not dst_rdir.exists():
                shutil.copytree(prev_rdir, dst_rdir)

    m_latest, meta_latest = load_model(prev / "latest.pt", "cpu")
    theta_latest = np.asarray(array(m_latest), dtype=np.float32)
    latest_digest = digest(theta_latest)
    if (
        int(meta_latest.get("round", -1)) != k_done
        or meta_latest.get("shared_pretrained_hash") != digest(study.initial)
        or meta_latest.get("policy_hash") != latest_digest
    ):
        raise ValueError(
            f"latest.pt round/hash mismatch: meta={meta_latest}, digest={latest_digest}"
        )
    cf_meta = study.history[-1].get("counterfactual_fixture")
    if isinstance(cf_meta, dict) and cf_meta.get("selected_sha256"):
        if latest_digest != cf_meta["selected_sha256"]:
            raise ValueError(
                "latest.pt policy_hash does not match final completed round selected_sha256"
            )
    val_last = study.history[-1].get("validation")
    if isinstance(val_last, dict) and val_last.get("policy_hash"):
        if latest_digest != val_last["policy_hash"]:
            raise ValueError(
                "latest.pt policy_hash does not match final round validation policy_hash"
            )
    study.center = theta_latest.copy()
    set_model(study.model, study.center)
    checkpoint(
        study.folder / "latest.pt",
        study.model_at(study.center),
        {
            "method": study.method,
            "round": k_done,
            "shared_pretrained_hash": digest(study.initial),
        },
    )

    m_best, meta_best = load_model(prev / "best-validation.pt", "cpu")
    theta_best = np.asarray(array(m_best), dtype=np.float32)
    best_digest = digest(theta_best)
    if (
        int(meta_best.get("selected_round", -1)) != best_round_expected
        or not np.isclose(
            float(meta_best.get("validation_mse", -1.0)),
            best_mse_expected,
            rtol=0.0,
            atol=1e-15,
        )
        or meta_best.get("shared_pretrained_hash") != digest(study.initial)
        or meta_best.get("policy_hash") != best_digest
    ):
        raise ValueError(
            f"best-validation.pt mismatch vs history: meta={meta_best}, digest={best_digest}, expected=({best_round_expected}, {best_mse_expected})"
        )
    if best_round_expected == 0:
        if best_digest != digest(study.initial):
            raise ValueError("best-validation.pt round 0 policy_hash does not match initial")
    elif 1 <= best_round_expected <= len(study.history):
        best_row = study.history[best_round_expected - 1]
        best_cf = best_row.get("counterfactual_fixture")
        if isinstance(best_cf, dict) and best_cf.get("selected_sha256"):
            if best_digest != best_cf["selected_sha256"]:
                raise ValueError(
                    "best-validation.pt policy_hash does not match selected round selected_sha256"
                )
        best_val = best_row.get("validation")
        if isinstance(best_val, dict) and best_val.get("policy_hash"):
            if best_digest != best_val["policy_hash"]:
                raise ValueError(
                    "best-validation.pt policy_hash does not match selected round validation policy_hash"
                )
    study.best = (best_mse_expected, best_round_expected, theta_best.copy())
    checkpoint(
        study.folder / "best-validation.pt",
        study.model_at(theta_best),
        {
            "method": study.method,
            "selected_round": best_round_expected,
            "validation_mse": best_mse_expected,
            "shared_pretrained_hash": digest(study.initial),
        },
    )
    set_model(study.model, study.center)

    prefix_win = 0
    prefix_out = 0
    aborted_win = 0
    aborted_out = 0
    aborted_prefix_a = f"main-r{k_done}-"
    aborted_prefix_b = f"round-{k_done + 1}-"
    for odir in shard_dirs:
        ledger_path = odir / "ledger.jsonl"
        if not ledger_path.is_file():
            continue
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            lbl = str(row.get("label", ""))
            c_win = int(row.get("cost", {}).get("window_attempts", 0))
            c_out = int(row.get("cost", {}).get("output_attempts", 0))
            if lbl.startswith(aborted_prefix_a) or lbl.startswith(aborted_prefix_b):
                aborted_win += c_win
                aborted_out += c_out
            else:
                prefix_win += c_win
                prefix_out += c_out

    study.cumulative_window_attempts = prefix_win
    study.cumulative_output_attempts = prefix_out
    study.elapsed_offset = float(study.history[-1].get("elapsed_seconds", 0.0))
    prev_status = (
        json.loads((prev / "status.json").read_text(encoding="utf-8"))
        if (prev / "status.json").is_file()
        else {}
    )
    prev_prog = (
        json.loads((prev / "progress.json").read_text(encoding="utf-8"))
        if (prev / "progress.json").is_file()
        else {}
    )
    study.milestones = dict(prev_prog.get("milestones", {}))
    if (prev / "checkpoints").is_dir() and not (study.folder / "checkpoints").exists():
        shutil.copytree(prev / "checkpoints", study.folder / "checkpoints")
    for m_ckpt in prev.glob("best-validation-*.pt"):
        dst_m = study.folder / m_ckpt.name
        if not dst_m.exists():
            shutil.copy2(m_ckpt, dst_m)
    study.resume_provenance = {
        "resumed_from_attempt": _relative_or_name(prev, study.folder),
        "resumed_from_initialization_sha256": file_digest(prev_init_path),
        "completed_rounds_restored": k_done,
        "restored_center_hash": digest(study.center),
        "restored_best_round": best_round_expected,
        "restored_best_validation_mse": best_mse_expected,
        "restored_best_hash": digest(theta_best),
        "completed_prefix_window_attempts": prefix_win,
        "completed_prefix_output_attempts": prefix_out,
        "aborted_next_round_ledgered_window_attempts": aborted_win,
        "aborted_next_round_ledgered_output_attempts": aborted_out,
        "previous_attempt_error": prev_status.get("error"),
    }
    save(study.folder / "resume-provenance.json", study.resume_provenance)
    save(
        study.folder / "progress.json",
        {
            "status": "training",
            "method": study.method,
            "rounds_completed": len(study.history),
            "last_round": k_done,
            "last_validation": next(
                (
                    {"round": r["round"], "mse": r["validation"]["mean_mse"]}
                    for r in reversed(study.history)
                    if "validation" in r
                ),
                None,
            ),
            "round_records": "rounds/round-*.json",
            "planned_rounds": study.recipe.fine_rounds,
            "resumed_from_round": k_done,
        },
    )
    return k_done, total_probe_retries


def update_best_checkpoint(study, k: int, val: dict, cur_win: int, row: dict) -> None:
    if not val["failures"] and val["mean_mse"] < study.best[0]:
        study.best = (val["mean_mse"], k, study.center.copy())
        study.best_score = val
        checkpoint(
            study.folder / "best-validation.pt",
            study.model_at(study.center),
            {
                "method": study.method,
                "selected_round": k,
                "validation_mse": val["mean_mse"],
                "cumulative_window_attempts": cur_win,
                "shared_pretrained_hash": digest(study.initial),
            },
        )
        ckpt_rel = f"checkpoints/best-round-{k:04d}.pt"
        checkpoint(
            study.folder / ckpt_rel,
            study.model_at(study.center),
            {
                "method": study.method,
                "selected_round": k,
                "validation_mse": val["mean_mse"],
                "cumulative_window_attempts": cur_win,
                "shared_pretrained_hash": digest(study.initial),
            },
        )
        row["new_best_checkpoint"] = ckpt_rel


def record_budget_exceeding_milestone(
    study, k: int, val: dict, budget: int, cur_win: int, cur_out: int, milestones: dict
) -> None:
    milestones["first_exceeding_window_budget"] = {
        "round": k,
        "target_window_budget": budget,
        "cumulative_window_attempts": cur_win,
        "cumulative_output_attempts": cur_out,
        "best_round": study.best[1],
        "best_validation_mse": study.best[0],
        "best_hash": digest(study.best[2]),
        "validation_at_round": val["mean_mse"],
        "checkpoint": "best-validation-exceeding-window-budget.pt",
    }
    checkpoint(
        study.folder / "best-validation-exceeding-window-budget.pt",
        study.model_at(study.best[2]),
        {
            "method": study.method,
            "milestone": "first_exceeding_window_budget",
            "milestone_round": k,
            "selected_round": study.best[1],
            "validation_mse": study.best[0],
            "target_window_budget": budget,
            "cumulative_window_attempts": cur_win,
            "shared_pretrained_hash": digest(study.initial),
        },
    )


def record_scheduled_milestones(
    study, k: int, val: dict, budget: int, cur_win: int, cur_out: int
) -> None:
    milestones = study.__dict__.setdefault("milestones", {})
    if study.stopped and "first_patience_stop" not in milestones:
        milestones["first_patience_stop"] = {
            "trigger_round": k,
            "patience": study.recipe.patience,
            "plateau_relative": study.recipe.plateau_relative,
            "best_round": study.best[1],
            "best_validation_mse": study.best[0],
            "best_hash": digest(study.best[2]),
            "cumulative_window_attempts": cur_win,
            "cumulative_output_attempts": cur_out,
            "checkpoint": "best-validation-first-patience-stop.pt",
        }
        checkpoint(
            study.folder / "best-validation-first-patience-stop.pt",
            study.model_at(study.best[2]),
            {
                "method": study.method,
                "milestone": "first_patience_stop",
                "trigger_round": k,
                "selected_round": study.best[1],
                "validation_mse": study.best[0],
                "cumulative_window_attempts": cur_win,
                "shared_pretrained_hash": digest(study.initial),
            },
        )
    if k == 100:
        milestones["round_100"] = {
            "round": k,
            "best_round": study.best[1],
            "best_validation_mse": study.best[0],
            "best_hash": digest(study.best[2]),
            "validation_at_round": val["mean_mse"],
            "cumulative_window_attempts": cur_win,
            "cumulative_output_attempts": cur_out,
            "checkpoint": "best-validation-round-100.pt",
        }
        checkpoint(
            study.folder / "best-validation-round-100.pt",
            study.model_at(study.best[2]),
            {
                "method": study.method,
                "milestone": "round_100",
                "milestone_round": k,
                "selected_round": study.best[1],
                "validation_mse": study.best[0],
                "cumulative_window_attempts": cur_win,
                "shared_pretrained_hash": digest(study.initial),
            },
        )
    if budget > 0:
        if cur_win <= budget:
            milestones["last_within_window_budget"] = {
                "round": k,
                "target_window_budget": budget,
                "cumulative_window_attempts": cur_win,
                "cumulative_output_attempts": cur_out,
                "best_round": study.best[1],
                "best_validation_mse": study.best[0],
                "best_hash": digest(study.best[2]),
                "validation_at_round": val["mean_mse"],
                "checkpoint": "best-validation-within-window-budget.pt",
            }
            checkpoint(
                study.folder / "best-validation-within-window-budget.pt",
                study.model_at(study.best[2]),
                {
                    "method": study.method,
                    "milestone": "last_within_window_budget",
                    "milestone_round": k,
                    "selected_round": study.best[1],
                    "validation_mse": study.best[0],
                    "target_window_budget": budget,
                    "cumulative_window_attempts": cur_win,
                    "shared_pretrained_hash": digest(study.initial),
                },
            )
        elif "first_exceeding_window_budget" not in milestones:
            record_budget_exceeding_milestone(study, k, val, budget, cur_win, cur_out, milestones)


def build_cost_checkpoints(study, pre_finish_windows: int) -> dict[str, Any] | None:
    anchor = int(getattr(study.recipe, "window_budget_anchor", 0) or 0)
    if anchor <= 0:
        return None
    init_mse = float(getattr(study, "initial_validation_mse", study.best[0]))
    init_hash = digest(study.initial)
    init_paid = int(getattr(study, "initial_paid_window_attempts", 0))
    validated_points = [
        {
            "round": 0,
            "paid_window_attempts": init_paid,
            "best_round": 0,
            "best_validation_mse": init_mse,
            "best_policy_hash": init_hash,
        }
    ]
    cur_best_mse = init_mse
    cur_best_round = 0
    cur_best_hash = init_hash
    for r in study.history:
        val = r.get("validation")
        if isinstance(val, dict) and not val.get("failures"):
            paid_w = int(r.get("cumulative_window_attempts", 0))
            v_mse = float(val["mean_mse"])
            if v_mse < cur_best_mse:
                cur_best_mse = v_mse
                cur_best_round = int(r["round"])
                cur_best_hash = str(val.get("policy_hash") or cur_best_hash)
            validated_points.append(
                {
                    "round": int(r["round"]),
                    "paid_window_attempts": paid_w,
                    "best_round": cur_best_round,
                    "best_validation_mse": cur_best_mse,
                    "best_policy_hash": cur_best_hash,
                }
            )

    ckpt_dir = study.folder / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    r0_ckpt = ckpt_dir / "best-round-0000.pt"
    if not r0_ckpt.is_file():
        checkpoint(
            r0_ckpt,
            study.model_at(study.initial),
            {
                "method": study.method,
                "selected_round": 0,
                "validation_mse": init_mse,
                "shared_pretrained_hash": init_hash,
            },
        )
        set_model(study.model, study.center)

    out: dict[str, Any] = {"anchor_window_attempts": anchor, "fractions": {}}
    for frac in (0.10, 0.25, 0.50, 1.00):
        tag = f"{int(round(frac * 100))}pct"
        cap_w = int(anchor * frac)
        eligible = [pt for pt in validated_points if pt["paid_window_attempts"] <= cap_w]
        chosen = eligible[-1] if eligible else validated_points[0]
        src_rel = f"checkpoints/best-round-{chosen['best_round']:04d}.pt"
        src_ckpt = study.folder / src_rel
        if not src_ckpt.is_file():
            raise FileNotFoundError(
                f"missing retained validation checkpoint for {tag} (round {chosen['best_round']}): {src_ckpt}"
            )
        m_chk, meta_chk = load_model(src_ckpt, "cpu")
        verified_hash = digest(array(m_chk))
        if (
            verified_hash != chosen["best_policy_hash"]
            or meta_chk.get("policy_hash") != verified_hash
        ):
            raise ValueError(
                f"budget checkpoint hash mismatch for {tag}: expected {chosen['best_policy_hash']}, got {verified_hash} (meta={meta_chk.get('policy_hash')})"
            )
        dst_rel = f"checkpoints/cost-{tag}.pt"
        dst_ckpt = study.folder / dst_rel
        shutil.copy2(src_ckpt, dst_ckpt)
        out["fractions"][tag] = {
            "fraction": frac,
            "threshold_window_attempts": cap_w,
            "reached_or_exceeded": int(pre_finish_windows) >= cap_w,
            "natural_stop_within_threshold": int(pre_finish_windows) <= cap_w,
            "last_validated_round_within_threshold": chosen["round"],
            "paid_window_attempts_at_checkpoint": chosen["paid_window_attempts"],
            "best_selected_round": chosen["best_round"],
            "best_validation_mse": chosen["best_validation_mse"],
            "best_policy_hash": chosen["best_policy_hash"],
            "checkpoint_path": dst_rel,
            "source_checkpoint_path": src_rel,
            "verified_policy_hash": verified_hash,
        }
    return out
