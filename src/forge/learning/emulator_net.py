"""Causal periodic NeuralEmulator architecture and open/closed-loop training shards."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from ..networks import SpectralConv2d
from ..paired.model import project_action
from .projection import parameter_slices


def build_callback_phase_table(intervals, output_after):
    """Construct purely public callback sequence features of shape (S, 4)."""
    intervals_arr = np.asarray(intervals, dtype=np.int64)
    after_arr = np.asarray(output_after, dtype=np.int64)
    steps = len(intervals_arr)
    table = np.zeros((steps, 4), dtype=np.float32)
    after_set = {int(x) for x in after_arr.tolist()}
    for k in range(int(intervals_arr.max()) + 1):
        idx = np.flatnonzero(intervals_arr == k)
        length = len(idx)
        denom = max(length - 1, 1)
        for j, tau in enumerate(idx.tolist()):
            table[tau, 0] = 1.0 if tau > 0 else 0.0
            table[tau, 1] = float(j) / float(denom)
            table[tau, 2] = float(j % 2)
            table[tau, 3] = 1.0 if tau in after_set else 0.0
    return table


def _interval_ranges(intervals, output_after):
    intervals_arr = np.asarray(intervals, dtype=np.int64)
    after_arr = np.asarray(output_after, dtype=np.int64)
    ranges = []
    for k, after_idx in enumerate(after_arr.tolist()):
        idx = np.flatnonzero(intervals_arr == k)
        if not len(idx) or int(idx[-1]) != int(after_idx):
            raise ValueError("public callback interval/output_after alignment mismatch")
        ranges.append((int(idx[0]), int(idx[-1]) + 1))
    return ranges


class _EmulatorBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, modes)
        self.local_conv = nn.Conv2d(width, width, kernel_size=3, padding=1, padding_mode="circular")
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        return x + F.gelu(self.spectral(x) + self.local_conv(x) + self.pointwise(x))


class NeuralEmulator(nn.Module):
    """Causal periodic sequence emulator for public callback observations and scoring fields."""

    def __init__(
        self,
        case_name,
        observation_channels,
        action_channels,
        field_channels,
        *,
        width=32,
        modes=8,
        layers=3,
        obs_mean=None,
        obs_std=None,
        field_std=None,
    ):
        super().__init__()
        self.case_name = case_name
        self.observation_channels = int(observation_channels)
        self.action_channels = int(action_channels)
        self.field_channels = int(field_channels)
        in_ch = 2 * self.observation_channels + 2 * self.action_channels + 4

        self.register_buffer(
            "obs_mean",
            torch.zeros(1, self.observation_channels, 1, 1)
            if obs_mean is None
            else torch.as_tensor(obs_mean, dtype=torch.float32).reshape(
                1, self.observation_channels, 1, 1
            ),
        )
        self.register_buffer(
            "obs_std",
            torch.ones(1, self.observation_channels, 1, 1)
            if obs_std is None
            else torch.as_tensor(obs_std, dtype=torch.float32)
            .reshape(1, self.observation_channels, 1, 1)
            .clamp(min=1e-6),
        )
        self.register_buffer(
            "field_std",
            torch.ones(1, self.field_channels, 1, 1)
            if field_std is None
            else torch.as_tensor(field_std, dtype=torch.float32)
            .reshape(1, self.field_channels, 1, 1)
            .clamp(min=1e-6),
        )

        self.lift = nn.Conv2d(in_ch, width, kernel_size=1)
        self.blocks = nn.ModuleList([_EmulatorBlock(width, modes) for _ in range(layers)])
        self.obs_head = nn.Conv2d(width, self.observation_channels, kernel_size=1)
        self.field_skip = nn.Conv2d(in_ch, self.field_channels, kernel_size=1)
        self.field_head = nn.Conv2d(width, self.field_channels, kernel_size=1)
        nn.init.zeros_(self.obs_head.weight)
        nn.init.zeros_(self.obs_head.bias)
        nn.init.zeros_(self.field_head.weight)
        nn.init.zeros_(self.field_head.bias)

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def denormalize_obs(self, obs_norm):
        return self.obs_mean + self.obs_std * obs_norm

    def step_callback(self, obs_curr, obs_prev, act_curr, act_prev, phase_vec, need_field=False):
        b, _, h, w = obs_curr.shape
        o_norm = self.normalize_obs(obs_curr)
        do_norm = (obs_curr - obs_prev) / self.obs_std
        phase_map = phase_vec.reshape(1, 4, 1, 1).expand(b, 4, h, w)
        inp = torch.cat((o_norm, do_norm, act_curr, act_prev, phase_map), dim=1)
        feat = F.gelu(self.lift(inp))
        for block in self.blocks:
            feat = block(feat)
        next_o_norm = o_norm + self.obs_head(feat)
        next_obs = self.denormalize_obs(next_o_norm)
        if not need_field:
            return next_obs, None
        pred_f_norm = self.field_skip(inp) + self.field_head(feat)
        pred_field = self.field_std * pred_f_norm
        return next_obs, pred_field

    def unroll_open_loop(
        self,
        obs_init,
        obs_prev_init,
        act_prev_init,
        actions_seq,
        phase_seq,
        output_rel_indices,
        obs_target_count=None,
    ):
        """Unroll E_psi over a contiguous callback segment given prescribed actions_seq."""
        obs_c = obs_init
        obs_p = obs_prev_init
        act_p = act_prev_init
        out_set = {int(i) for i in output_rel_indices}
        obs_preds = []
        field_preds = []
        steps = actions_seq.shape[0]
        keep_obs = (steps - 1) if obs_target_count is None else int(obs_target_count)
        for t in range(steps):
            act_c = actions_seq[t]
            need_f = t in out_set
            next_obs, pred_f = self.step_callback(
                obs_c, obs_p, act_c, act_p, phase_seq[t], need_field=need_f
            )
            if t < keep_obs:
                obs_preds.append(next_obs)
            if need_f:
                field_preds.append(pred_f)
            obs_p = obs_c
            obs_c = next_obs
            act_p = act_c
        return obs_preds, field_preds


class _EmulatorFitShard(nn.Module):
    """Outer module wrapped by DDP once for multi-step open-loop + perturbation-diff training."""

    def __init__(self, emulator):
        super().__init__()
        self.emulator = emulator

    def forward(
        self,
        obs0_init,
        obs0_prev,
        act0_prev,
        act0_seq,
        target_obs0,
        target_field0,
        obsp_init,
        obsp_prev,
        actp_prev,
        actp_seq,
        target_obsp,
        target_fieldp,
        phase_seq,
        output_rel_indices,
        diff_scale,
    ):
        n_obs_targets = target_obs0.shape[0]

        def _run_pair(o_i, o_p, a_p, a_seq):
            obs_list, f_list = self.emulator.unroll_open_loop(
                o_i,
                o_p,
                a_p,
                a_seq,
                phase_seq,
                output_rel_indices,
                obs_target_count=n_obs_targets,
            )
            o_stack = torch.stack(obs_list, dim=0) if obs_list else o_i.new_zeros((0, *o_i.shape))
            f_stack = torch.stack(f_list, dim=0)
            return o_stack, f_stack

        pred_o0, pred_f0 = _run_pair(obs0_init, obs0_prev, act0_prev, act0_seq)
        pred_op, pred_fp = _run_pair(obsp_init, obsp_prev, actp_prev, actp_seq)

        obs_std = self.emulator.obs_std.unsqueeze(0)
        f_std = self.emulator.field_std.unsqueeze(0)

        has_obs = n_obs_targets > 0
        loss_obs = (
            0.5
            * (
                ((pred_o0 - target_obs0) / obs_std).square().mean()
                + ((pred_op - target_obsp) / obs_std).square().mean()
            )
            if has_obs
            else pred_f0.new_zeros(())
        )
        loss_field = 0.5 * (
            ((pred_f0 - target_field0) / f_std).square().mean()
            + ((pred_fp - target_fieldp) / f_std).square().mean()
        )

        inv_s = 1.0 / max(float(diff_scale), 1e-3)
        df_pred = ((pred_fp - pred_f0) / f_std) * inv_s
        df_true = ((target_fieldp - target_field0) / f_std) * inv_s
        if has_obs:
            do_pred = ((pred_op - pred_o0) / obs_std) * inv_s
            do_true = ((target_obsp - target_obs0) / obs_std) * inv_s
            loss_diff = 0.5 * (
                (do_pred - do_true).square().mean() + (df_pred - df_true).square().mean()
            )
        else:
            loss_diff = (df_pred - df_true).square().mean()
        return loss_obs + loss_field + 0.5 * loss_diff


class _EmulatorPolicyShard(nn.Module):
    """Outer module wrapped by DDP once for closed-loop policy optimization through frozen E_psi."""

    def __init__(
        self,
        actor_model,
        emulator,
        case_name,
        intervals,
        output_after,
        start_vec,
        sigma_action,
    ):
        super().__init__()
        self.model = actor_model
        self.emulator = emulator
        self.case_name = case_name
        self.sigma_action = float(sigma_action)
        self.ranges = _interval_ranges(intervals, output_after)
        self.register_buffer(
            "start_vec",
            torch.as_tensor(start_vec, dtype=torch.float32).detach().clone(),
        )
        self.register_buffer(
            "phase_table",
            torch.as_tensor(
                build_callback_phase_table(intervals, output_after), dtype=torch.float32
            ),
        )

    def forward(self, initial_obs, reference, base_initial_actions=None):
        obs_c = initial_obs
        obs_p = initial_obs
        b, _, h, w = initial_obs.shape
        act_p = initial_obs.new_zeros((b, self.model.action_channels, h, w))
        f_std = self.emulator.field_std
        total_mse = initial_obs.new_zeros(())
        act_sq_total = initial_obs.new_zeros(())
        track_act_rms = not torch.is_grad_enabled()

        first_act = project_action(self.case_name, self.model(initial_obs))
        da0_sq = (
            (first_act - base_initial_actions).square().mean()
            if base_initial_actions is not None
            else initial_obs.new_zeros(())
        )

        for k, (start_t, end_t) in enumerate(self.ranges):
            phase_slice = self.phase_table[start_t:end_t]

            if torch.is_grad_enabled():

                def _step_interval_grad(o_c, o_p, a_p, p_slice=phase_slice):
                    cur_o = o_c
                    prev_o = o_p
                    prev_a = a_p
                    f_out = None
                    length = p_slice.shape[0]
                    for j in range(length):
                        act_a = project_action(self.case_name, self.model(cur_o))
                        need_f = j == length - 1
                        nxt_o, p_f = self.emulator.step_callback(
                            cur_o, prev_o, act_a, prev_a, p_slice[j], need_field=need_f
                        )
                        if need_f:
                            f_out = p_f
                        prev_o = cur_o
                        cur_o = nxt_o
                        prev_a = act_a
                    return cur_o, prev_o, prev_a, f_out

                obs_c, obs_p, act_p, pred_fk = grad_checkpoint(
                    _step_interval_grad,
                    obs_c,
                    obs_p,
                    act_p,
                    use_reentrant=False,
                )
            else:
                cur_o, prev_o, prev_a = obs_c, obs_p, act_p
                a_sq = obs_c.new_zeros(())
                pred_fk = None
                length = phase_slice.shape[0]
                for j in range(length):
                    act_a = project_action(self.case_name, self.model(cur_o))
                    if track_act_rms:
                        a_sq = a_sq + act_a.square().mean()
                    need_f = j == length - 1
                    nxt_o, p_f = self.emulator.step_callback(
                        cur_o,
                        prev_o,
                        act_a,
                        prev_a,
                        phase_slice[j],
                        need_field=need_f,
                    )
                    if need_f:
                        pred_fk = p_f
                    prev_o = cur_o
                    cur_o = nxt_o
                    prev_a = act_a
                obs_c, obs_p, act_p = cur_o, prev_o, prev_a
                act_sq_total = act_sq_total + (a_sq / max(length, 1)) / float(len(self.ranges))

            err_k = ((pred_fk - reference[:, k]) / f_std).square().mean(dim=(1, 2, 3))
            if not torch.isfinite(err_k).all():
                raise FloatingPointError("nonfinite emulator closed-loop rollout")
            capped = err_k.clamp(max=200.0)
            total_mse = total_mse + capped.mean() / float(len(self.ranges))

        parameter_trust_penalty = 1e-3 * self.parameter_trust_term()
        initial_action_trust_penalty = self.initial_action_trust_term(da0_sq)
        total_loss = total_mse + parameter_trust_penalty + initial_action_trust_penalty
        return total_loss, total_mse.detach(), act_sq_total.detach(), da0_sq.detach()

    def parameter_trust_term(self):
        return sum(
            (param.reshape(-1) - self.start_vec[sl]).square().sum()
            / self.start_vec[sl].square().sum().clamp(min=1.0)
            for param, sl in parameter_slices(self.model)
        )

    def initial_action_trust_term(self, da0_sq):
        return F.relu(da0_sq - (2.0 * self.sigma_action) ** 2)


__all__ = [
    "NeuralEmulator",
    "_EmulatorBlock",
    "_EmulatorFitShard",
    "_EmulatorPolicyShard",
    "_interval_ranges",
    "build_callback_phase_table",
]
