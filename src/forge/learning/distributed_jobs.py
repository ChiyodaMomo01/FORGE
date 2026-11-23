"""Collective jobs on public arrays and owned networks, never F/H internals."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from ..core.io import load_array
from .client import Client, merge_scores
from .closed_loop import fit_response_job
from .closure_rl import closure_rl_ddp_job
from .common import digest, load_model, set_model
from .distributed import full_prediction, rank, size, wrap
from .emulator import emulator_ddp_job
from .policy import SpatialCritic
from .ppo_joint import actor_step, finalize_candidate_kl
from .projection import project as project_actions
from .response_data import ResponseTrace
from .subspace import validate_paired_coordinates
from .tangent.traces import IntervalTrace


def setup(e, p):
    close_client(e, None)
    model, _ = load_model(p["checkpoint"], e.device)
    if getattr(p["recipe"], "actor_fp64", False):
        model = model.double()
    e.state["query_client"] = Client(
        p["root"],
        Path(p["folder"]) / f"gpu-{rank()}",
        str(e.device),
        p["recipe"],
        p["checkpoint"],
        external_model=model,
    )
    e.state["checkpoint"] = p["checkpoint"]
    return e.state["query_client"].info


def close_client(e, p):
    del p
    if "query_client" in e.state:
        e.state.pop("query_client").close()


def query(e, p):
    client = e.state["query_client"]
    theta_f32 = np.asarray(p["theta"], dtype=np.float32)
    local_hash = digest(theta_f32)
    selected = (
        list(range(client.info["trials"][p["split"]])) if p["trials"] is None else list(p["trials"])
    )
    part = selected[rank() :: size()]
    value = client.score(theta_f32, p["split"], part, **p["kwargs"]) if part else None
    rows = e.gather((local_hash, value))
    if rank() == 0:
        hashes = [h for h, _ in rows]
        if any(h != hashes[0] for h in hashes):
            raise RuntimeError("distributed policy-hash parity mismatch across ranks")
        shard_rows = [r for _, r in rows if r is not None]
        if any(
            r.get("policy_hash") is not None and r.get("policy_hash") != hashes[0]
            for r in shard_rows
        ):
            raise RuntimeError("distributed shard policy-hash mismatch")
        merged = merge_scores(shard_rows, selected, p["kwargs"])
        merged["policy_hash"] = hashes[0]
        merged["policy_hash_parity"] = {
            "verified": True,
            "ranks": len(hashes),
            "policy_hash": hashes[0],
        }
        return merged
    return None


def residual(e, p):
    z = np.asarray(p["coordinates"], np.float64)
    paired_radii = p.get("paired_radii")
    r = validate_paired_coordinates(z, paired_radii) if paired_radii is not None else None
    if len(p["member_paths"]) != len(z):
        raise ValueError("member_paths count must match coordinate rows")
    d = z.shape[1]
    pinv = torch.as_tensor(np.linalg.pinv(z, rcond=1e-10), device=e.device)
    gram = torch.zeros((d, d), dtype=torch.float64, device=e.device)
    cross = torch.zeros(d, dtype=torch.float64, device=e.device)
    pair_stats = (
        torch.zeros((5, d), dtype=torch.float64, device=e.device) if r is not None else None
    )
    count = torch.zeros((), dtype=torch.float64, device=e.device)
    for part in range(rank(), len(p["center_paths"]), size()):
        center = load_array(p["center_paths"][part], mmap_mode="r").reshape(-1)
        members = [
            load_array(paths[part], mmap_mode="r").reshape(-1) for paths in p["member_paths"]
        ]
        if any(a.shape != center.shape for a in members):
            raise ValueError("unaligned public residuals")
        for start in range(0, len(center), 65536):
            c = torch.as_tensor(
                np.array(center[start : start + 65536], dtype=np.float64),
                device=e.device,
            )
            values = torch.as_tensor(
                np.stack([a[start : start + 65536] for a in members]),
                dtype=torch.float64,
                device=e.device,
            )
            differences = values - c
            response_vec = pinv @ differences
            gram += response_vec @ response_vec.T
            cross += response_vec @ c
            if pair_stats is not None:
                diff_plus = differences[:d]
                diff_minus = differences[d:]
                odd = 0.5 * (diff_plus - diff_minus)
                even = 0.5 * (diff_plus + diff_minus)
                pair_stats[0] += torch.sum(even * even, dim=1)
                pair_stats[1] += torch.sum(odd * odd, dim=1)
                pair_stats[2] += torch.sum(diff_plus * diff_plus, dim=1)
                pair_stats[3] += torch.sum(diff_minus * diff_minus, dim=1)
                pair_stats[4] += torch.sum(diff_plus * (-diff_minus), dim=1)
            count += len(c)
    for value in (gram, cross, count):
        dist.all_reduce(value)
    if pair_stats is not None:
        dist.all_reduce(pair_stats)
    total_count = count.item()
    meta = {
        "coordinate_rank": int(np.linalg.matrix_rank(z)),
        "residual_count": int(total_count),
        "coordinate_dimension": len(gram),
        "centered_on": "deterministic_policy",
        "distributed_ranks": size(),
    }
    if pair_stats is not None:
        ps = pair_stats.cpu().numpy()
        e_diag = ps[0] / total_count
        h_diag = ps[1] / total_count
        plus_sq = ps[2]
        minus_sq = ps[3]
        anti_dot = ps[4]
        meta.update(
            {
                "E_diag_fd": e_diag.tolist(),
                "H_diag_fd": h_diag.tolist(),
                "E_coordinate_convention": "dimensionless_probe_coordinates_z",
                "fd_even_over_odd_rms": np.sqrt(e_diag / np.maximum(h_diag, 1e-30)).tolist(),
                "fd_antithetic_cosine": (
                    anti_dot / np.sqrt(np.maximum(plus_sq * minus_sq, 1e-30))
                ).tolist(),
            }
        )
    return (
        gram.cpu().numpy() / total_count,
        cross.cpu().numpy() / total_count,
        meta,
    )


def project(e, p):
    """Run the shared projection on the coordinator; workers await the next job."""
    if rank() != 0:
        return None
    model, _ = load_model(e.state["checkpoint"], e.device)
    traces = [IntervalTrace(ResponseTrace(path)) for path in p["paths"]]
    options = {
        k: p[k]
        for k in (
            "updates",
            "batch_pairs",
            "micro_pairs",
            "seed",
            "l2_only",
            "head_only",
        )
        if k in p
    }
    return project_actions(model, p["theta"], traces, p["corrections"], **options)


def ppo(e, p):
    recipe = p["recipe"]
    if "ppo" not in e.state:
        model, _ = load_model(e.state["checkpoint"], e.device)
        channels = recipe.observation_channels
        critic = SpatialCritic(
            channels,
            16,
            model.network.input_mean[:, :channels],
            model.network.input_std[:, :channels],
        ).to(e.device)
        critic.load_state_dict(p["critic"])
        model = model.double()
        e.state["ppo"] = [
            model,
            critic,
            wrap(model),
            wrap(critic),
            torch.optim.Adam(model.parameters(), lr=recipe.ppo_lr),
            torch.optim.Adam(critic.parameters(), lr=3e-4),
        ]
    state = e.state["ppo"]
    model, critic, actor_forward, critic_forward, actor_opt, critic_opt = state
    set_model(model, p["theta"])
    if p["reset_optimizer"]:
        actor_opt = state[4] = torch.optim.Adam(model.parameters(), lr=recipe.ppo_lr)
    obs, actions, means, times, returns = [np.load(path, mmap_mode="r") for path in p["arrays"]]
    global_credit = p["global_credit"]
    local, group = not global_credit, 4 if global_credit else 1
    values = []
    for begin in range(0, len(obs), 256):
        n = min(256, len(obs) - begin)
        ids = np.arange(rank(), n, size())
        x = torch.as_tensor(np.array(obs[begin + ids]), device=e.device)
        with torch.no_grad():
            v = critic(
                x,
                torch.as_tensor(np.array(times[begin + ids]), device=e.device),
                recipe.horizon,
            )
            v = v if local else v.mean((1, 2), keepdim=True)
        pieces = e.gather((ids, v.cpu().numpy()))
        if rank() == 0:
            combined = np.empty((n, *v.shape[1:]), dtype=np.float32)
            for ix, value in pieces:
                combined[ix] = value
            values.extend(combined)
    advantage = None
    if rank() == 0:
        advantage = returns - np.asarray(values)
        if global_credit:
            advantage = advantage.reshape(-1, 4, 1, 1).mean((1, 2, 3))
        advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-8)
    advantage = e.broadcast(advantage)
    rng = np.random.default_rng(820000 + p["round"])
    units = len(obs) // group
    checks, stopped = [], False

    for _ in range(recipe.ppo_epochs):
        for unit_ids in np.array_split(
            rng.permutation(units), math.ceil(units / max(1, 8 // group))
        ):
            ix = (unit_ids[:, None] * group + np.arange(group)).reshape(-1)
            x = torch.as_tensor(np.array(obs[ix]), device=e.device)
            if not stopped:
                x_actor = x.double()
                check = actor_step(
                    model,
                    actor_opt,
                    x_actor,
                    torch.as_tensor(np.array(actions[ix]), device=e.device, dtype=torch.float64),
                    torch.as_tensor(np.array(means[ix]), device=e.device, dtype=torch.float64),
                    torch.as_tensor(advantage[unit_ids], device=e.device, dtype=torch.float64),
                    recipe.sigma_action,
                    group,
                    local,
                    forward_model=lambda data: full_prediction(actor_forward, data),
                )
                checks.append(check)
                stopped = bool(check.get("stop_epoch", not check["updated"]))
            n = len(ix)
            width = (n + size() - 1) // size()
            local_ids = np.minimum(np.arange(rank() * width, (rank() + 1) * width), n - 1)
            v = critic_forward(
                x[local_ids],
                torch.as_tensor(np.array(times[ix[local_ids]]), device=e.device),
                recipe.horizon,
            )
            value = torch.cat(all_gather(v))[:n]
            if global_credit:
                value = value.mean((1, 2), keepdim=True)
            loss = (value - torch.as_tensor(np.array(returns[ix]), device=e.device)).square().mean()
            critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            critic_opt.step()
    proposed, full_kl_check = finalize_candidate_kl(
        model,
        p["theta"],
        obs,
        actions,
        means,
        recipe.sigma_action,
        group,
        local,
        e.device,
        forward_model=lambda data: full_prediction(actor_forward, data),
    )
    checks.append({"full_dataset_kl_check": full_kl_check})
    return proposed, checks


def response(e, p):
    """The coordinator fits the same public-array model as serial execution."""
    if rank() != 0:
        return None
    model, _ = load_model(e.state["checkpoint"], e.device)
    return fit_response_job(model, p)


def closure_rl(e, p):
    return closure_rl_ddp_job(e, p)


def emulator_step(e, p):
    return emulator_ddp_job(e, p)


HANDLERS = {
    "setup": setup,
    "query": query,
    "close_client": close_client,
    "residual": residual,
    "project": project,
    "ppo": ppo,
    "response": response,
    "closure_rl": closure_rl,
    "emulator_step": emulator_step,
}
