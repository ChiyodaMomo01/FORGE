"""Public trace-pack extraction, unseen perturbation evaluation, and DDP buffer serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .common import save
from .emulator_net import build_callback_phase_table
from .response_data import ResponseTrace


def extract_emulator_trace_pack(center_paths, pert_scores):
    """Extract ONLY public coarse-solver arrays (never HR reference.npy) for E_psi fitting."""
    centers = {}
    for p in center_paths:
        tr = ResponseTrace(p)
        trial = int(tr.meta["trial"])
        centers[trial] = {
            "trial": trial,
            "observations": np.asarray(tr.observations, dtype=np.float32),
            "actions": np.asarray(tr.actions, dtype=np.float32),
            "fields": np.asarray(tr.fields, dtype=np.float32),
            "intervals": np.asarray(tr.intervals, dtype=np.int64),
            "output_after": np.asarray(tr.output_after, dtype=np.int64),
        }
    trials_sorted = sorted(centers.keys())
    if not trials_sorted:
        raise ValueError("no center public traces provided for emulator fitting")
    first = centers[trials_sorted[0]]
    intervals = first["intervals"]
    output_after = first["output_after"]

    c_obs = np.concatenate([centers[t]["observations"] for t in trials_sorted], axis=1)
    c_act = np.concatenate([centers[t]["actions"] for t in trials_sorted], axis=1)
    c_fld = np.concatenate([centers[t]["fields"] for t in trials_sorted], axis=1)

    pert_packs = []
    for p_info in pert_scores:
        sigma_val = abs(float(p_info["sigma"]))
        by_trial = {}
        for p in p_info["trace_paths"]:
            tr = ResponseTrace(p)
            by_trial[int(tr.meta["trial"])] = (
                np.asarray(tr.observations, dtype=np.float32),
                np.asarray(tr.actions, dtype=np.float32),
                np.asarray(tr.fields, dtype=np.float32),
            )
        if sorted(by_trial.keys()) != trials_sorted:
            raise ValueError("perturbed trace trials do not match center trace trials")
        pert_packs.append(
            {
                "sigma": max(sigma_val, 1e-3),
                "observations": np.concatenate([by_trial[t][0] for t in trials_sorted], axis=1),
                "actions": np.concatenate([by_trial[t][1] for t in trials_sorted], axis=1),
                "fields": np.concatenate([by_trial[t][2] for t in trials_sorted], axis=1),
            }
        )
    return {
        "trials": trials_sorted,
        "intervals": intervals,
        "output_after": output_after,
        "center_obs": c_obs,
        "center_act": c_act,
        "center_fields": c_fld,
        "perturbations": pert_packs,
    }


def extract_policy_task_pack(center_paths):
    """Extract training initial observations and HR task references exclusively for G_theta."""
    by_trial = {}
    for p in center_paths:
        tr = ResponseTrace(p)
        task = tr.task()
        by_trial[int(tr.meta["trial"])] = (
            np.asarray(task["initial_observation"], dtype=np.float32),
            np.asarray(task["reference"], dtype=np.float32),
            np.asarray(tr.actions[0], dtype=np.float32),
            np.asarray(task["field_std"], dtype=np.float32),
            np.asarray(tr.intervals, dtype=np.int64),
            np.asarray(task["output_after_sequence"], dtype=np.int64),
        )
    trials_sorted = sorted(by_trial.keys())
    first = by_trial[trials_sorted[0]]
    return {
        "initial_obs": np.concatenate([by_trial[t][0] for t in trials_sorted], axis=0),
        "reference": np.concatenate([by_trial[t][1] for t in trials_sorted], axis=0),
        "base_initial_act": np.concatenate([by_trial[t][2] for t in trials_sorted], axis=0),
        "field_std": first[3],
        "intervals": first[4],
        "output_after": first[5],
    }


@torch.no_grad()
def evaluate_emulator_unseen(emulator, held_pack, device):
    """Full-window rollout evaluation on unseen action perturbations."""
    emulator.eval()
    intervals = held_pack["intervals"]
    output_after = held_pack["output_after"]
    phase = torch.as_tensor(
        build_callback_phase_table(intervals, output_after),
        dtype=torch.float32,
        device=device,
    )
    c_obs0 = torch.as_tensor(held_pack["center_obs"][0], device=device)
    c_act = torch.as_tensor(held_pack["center_act"], device=device)
    c_fld_true = torch.as_tensor(held_pack["center_fields"], device=device)
    b, _, h, w = c_obs0.shape
    zero_act = c_obs0.new_zeros((b, emulator.action_channels, h, w))

    _, c_fld_pred_list = emulator.unroll_open_loop(
        c_obs0, c_obs0, zero_act, c_act, phase, output_after.tolist()
    )
    c_fld_pred = torch.stack(c_fld_pred_list, dim=0)
    f_std = emulator.field_std.unsqueeze(0)

    probe_rres = []
    probe_cosines = []
    probe_field_mses = []
    total_err_sq = 0.0
    total_true_sq = 0.0
    total_dot = 0.0
    total_pred_sq = 0.0

    for pert in held_pack["perturbations"]:
        p_obs0 = torch.as_tensor(pert["observations"][0], device=device)
        p_act = torch.as_tensor(pert["actions"], device=device)
        p_fld_true = torch.as_tensor(pert["fields"], device=device)
        _, p_fld_pred_list = emulator.unroll_open_loop(
            p_obs0, p_obs0, zero_act, p_act, phase, output_after.tolist()
        )
        p_fld_pred = torch.stack(p_fld_pred_list, dim=0)

        fld_mse = float(((p_fld_pred - p_fld_true) / f_std).square().mean().item())
        probe_field_mses.append(fld_mse)

        df_pred = ((p_fld_pred - c_fld_pred) / f_std).double()
        df_true = ((p_fld_true - c_fld_true) / f_std).double()
        err_sq = float((df_pred - df_true).square().sum().item())
        true_sq = float(df_true.square().sum().item())
        pred_sq = float(df_pred.square().sum().item())
        dot = float((df_pred * df_true).sum().item())

        rre = math.sqrt(err_sq / max(true_sq, 1e-24))
        cos = dot / max(math.sqrt(pred_sq * true_sq), 1e-24)
        probe_rres.append(rre)
        probe_cosines.append(cos)
        total_err_sq += err_sq
        total_true_sq += true_sq
        total_pred_sq += pred_sq
        total_dot += dot

    center_field_mse = float(((c_fld_pred - c_fld_true) / f_std).square().mean().item())
    return {
        "center_full_window_field_mse": center_field_mse,
        "held_full_window_field_mse_mean": (
            float(np.mean(probe_field_mses)) if probe_field_mses else 0.0
        ),
        "unseen_diff_rre": math.sqrt(total_err_sq / max(total_true_sq, 1e-24)),
        "unseen_diff_rre_per_probe": probe_rres,
        "unseen_diff_rre_worst": float(max(probe_rres)) if probe_rres else 0.0,
        "unseen_diff_cosine": total_dot / max(math.sqrt(total_pred_sq * total_true_sq), 1e-24),
        "unseen_diff_cosine_per_probe": probe_cosines,
    }


def _serialize_pack(pack, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "intervals.npy", pack["intervals"])
    np.save(folder / "output_after.npy", pack["output_after"])
    np.save(folder / "center_obs.npy", pack["center_obs"])
    np.save(folder / "center_act.npy", pack["center_act"])
    np.save(folder / "center_fields.npy", pack["center_fields"])
    sigmas = []
    for idx, p in enumerate(pack["perturbations"]):
        sigmas.append(float(p["sigma"]))
        np.save(folder / f"pert_{idx}_obs.npy", p["observations"])
        np.save(folder / f"pert_{idx}_act.npy", p["actions"])
        np.save(folder / f"pert_{idx}_fields.npy", p["fields"])
    save(
        folder / "meta.json",
        {"trials": pack["trials"], "sigmas": sigmas},
    )
    return str(folder)


def _deserialize_pack(folder):
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    perts = []
    for idx, sig in enumerate(meta["sigmas"]):
        perts.append(
            {
                "sigma": float(sig),
                "observations": np.load(folder / f"pert_{idx}_obs.npy", mmap_mode="r"),
                "actions": np.load(folder / f"pert_{idx}_act.npy", mmap_mode="r"),
                "fields": np.load(folder / f"pert_{idx}_fields.npy", mmap_mode="r"),
            }
        )
    return {
        "trials": meta["trials"],
        "intervals": np.load(folder / "intervals.npy"),
        "output_after": np.load(folder / "output_after.npy"),
        "center_obs": np.load(folder / "center_obs.npy", mmap_mode="r"),
        "center_act": np.load(folder / "center_act.npy", mmap_mode="r"),
        "center_fields": np.load(folder / "center_fields.npy", mmap_mode="r"),
        "perturbations": perts,
    }


__all__ = [
    "_deserialize_pack",
    "_serialize_pack",
    "evaluate_emulator_unseen",
    "extract_emulator_trace_pack",
    "extract_policy_task_pack",
]
