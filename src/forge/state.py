"""Device-neutral opaque-state helpers; integrator histories stay with the case."""

import copy

import torch


def tree_map(fn, value):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: tree_map(fn, v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_map(fn, v) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    return copy.deepcopy(value)


def to_device(state, device, dtype=None):
    def convert(x):
        target = (
            (torch.complex128 if dtype == torch.float64 else torch.complex64)
            if x.is_complex() and dtype
            else dtype
            if x.is_floating_point()
            else None
        )
        return x.detach().to(device=device, dtype=target).clone()

    return tree_map(convert, state)


def select(state, index, batch):
    return tree_map(
        lambda x: (
            x[index : index + 1].detach().cpu().clone()
            if x.ndim and x.shape[0] == batch
            else x.detach().cpu().clone()
        ),
        state,
    )


def concatenate(states):
    x = states[0]
    if isinstance(x, torch.Tensor):
        return torch.cat(states, dim=0) if x.ndim else x
    if isinstance(x, dict):
        return {k: concatenate([s[k] for s in states]) for k in x}
    if isinstance(x, list):
        return [concatenate([s[k] for s in states]) for k in range(len(x))]
    return copy.deepcopy(x)


def advance(case, state, dt, model=None):
    import math

    n = case.fields(state).shape[-1]
    substeps = max(1, math.ceil(dt / case.default_dt(n) - 1e-10))
    for _ in range(substeps):
        state = case.step(state, model=model, dt=dt / substeps)
    return state
