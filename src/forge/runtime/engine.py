"""Trusted supervised/AD trajectory engine. Strict learners never import this."""

import torch
from torch.utils.checkpoint import checkpoint

from forge.cases import make_case
from forge.state import advance


def case_for(bank, split):
    return make_case(bank.recipe.case_name, bank.info["splits"][split]["deployment_config"])


def trajectory_loss(case, initial, target, model, scale, dt=0.1, backward=False):
    def physical(obs):
        return model.physical(obs)

    physical.output_semantics = "full-g-v1"
    state = case.state_from_fields(initial)
    loss = initial.new_zeros(())
    scale = initial.new_tensor(scale).reshape(1, -1, 1, 1)
    for k in range(target.shape[1]):

        def step(s):
            return advance(case, s, dt, physical)

        state = checkpoint(step, state, use_reentrant=False) if backward else step(state)
        error = ((case.fields(state) - target[:, k]) / scale).square().mean((1, 2, 3))
        if not torch.isfinite(error).all():
            raise FloatingPointError("nonfinite window trajectory")
        loss += error.clamp(max=200).mean() / target.shape[1]
    return loss
