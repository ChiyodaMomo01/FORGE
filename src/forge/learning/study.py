"""Shared-pretrained public learner state; validation alone selects checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..core.io import sanitize_persisted_paths
from .common import actor, checkpoint, digest, save, set_model
from .events import anchors
from .numeric import configure
from .schema import RECIPE
from .study_state import (
    build_cost_checkpoints,
    record_budget_exceeding_milestone,
    record_scheduled_milestones,
    resume_study_from_attempt,
    update_best_checkpoint,
)


class Study:
    def __init__(
        self,
        client,
        folder,
        method,
        device,
        recipe=RECIPE,
        reference=None,
    ):
        self.actor_numeric = configure(recipe)
        if recipe.strict_actor_fp32:
            info = client.info.get("actor_numeric", {})
            if (
                info.get("cudnn_allow_tf32") is not False
                or info.get("matmul_allow_tf32") is not False
            ):
                raise ValueError("strict learner/value-server actor precision mismatch")
        self.client = client
        self.folder = Path(folder)
        self.method = method
        self.recipe = recipe
        self.started = time.monotonic()
        self.model = actor(
            client.info["n"],
            client.info["width"],
            device,
            recipe.case_name,
            statistics=client.info["statistics"],
        )
        self.initial = np.asarray(client.info["theta"], np.float32)
        self.center = self.initial.copy()
        set_model(self.model, self.center)
        self.history = []
        self.best = (float("inf"), 0, self.center.copy())
        self.train = None
        self._order = np.array([], dtype=int)
        self._epoch = 0
        self.reference = Path(reference) if reference else None
        self.significant_best = float("inf")
        self.bad_checks = 0
        self.stopped = False
        self.milestones = {}
        self.cumulative_window_attempts = 0
        self.cumulative_output_attempts = 0
        self.elapsed_offset = 0.0
        self.resume_provenance = None
        save(
            self.folder / "initialization.json",
            {
                "policy_hash": digest(self.initial),
                "source": "common.pt",
                "shared_pretrained": client.info.get("initialization") == "paired_pretrain",
                "initialization": client.info.get("initialization", "paired_pretrain"),
                "model_parameters": len(self.initial),
                "actor_forward_dtype": (
                    "float64" if getattr(self.recipe, "actor_fp64", False) else "float32"
                ),
            },
        )

    def initialize(self):
        if self.reference:
            retained = json.loads(
                (self.reference / "common/result.json").read_text(encoding="utf-8")
            )
            if retained["policy_hash"] != digest(self.initial):
                raise ValueError("retained common policy mismatch")
            self.train = retained["training"]
            val = retained["evaluations"]["validation"]
            if getattr(self.recipe, "actor_fp64", False):
                val_fp64 = self.score(
                    self.center,
                    "validation",
                    None,
                    label="shared-pretrained-validation-fp64",
                )
                save(
                    self.folder / "initialization.json",
                    {
                        "policy_hash": digest(self.initial),
                        "source": "common.pt",
                        "shared_pretrained": self.client.info.get("initialization")
                        == "paired_pretrain",
                        "initialization": self.client.info.get("initialization", "paired_pretrain"),
                        "model_parameters": len(self.initial),
                        "actor_forward_dtype": "float64",
                        "retained_fp32_validation_mse": val["mean_mse"],
                        "fp64_validation_mse": val_fp64["mean_mse"],
                        "fp64_validation": val_fp64,
                    },
                )
                val = val_fp64
        else:
            self.train = self.score(self.center, "train", None, label="shared-pretrained-train")
            val = self.score(self.center, "validation", None, label="shared-pretrained-validation")
        self.best = (val["mean_mse"], 0, self.center.copy())
        self.best_score = val
        self.initial_validation_mse = float(val["mean_mse"])
        self.significant_best = val["mean_mse"]
        checkpoint(
            self.folder / "best-validation.pt",
            self.model_at(self.center),
            {
                "method": self.method,
                "selected_round": 0,
                "validation_mse": val["mean_mse"],
                "shared_pretrained_hash": digest(self.initial),
            },
        )
        checkpoint(
            self.folder / "checkpoints" / "best-round-0000.pt",
            self.model_at(self.center),
            {
                "method": self.method,
                "selected_round": 0,
                "validation_mse": val["mean_mse"],
                "cumulative_window_attempts": self.cumulative_window_attempts,
                "shared_pretrained_hash": digest(self.initial),
            },
        )
        logged = self.score(
            self.center,
            "train",
            self.batch(),
            capacity=1,
            label="pretrained-public-anchors",
            seed=2000,
        )
        self.initial_paid_window_attempts = getattr(self, "cumulative_window_attempts", 0)
        self.obs = anchors(logged, 16)
        budget = int(getattr(self.recipe, "target_window_budget", 0))
        if budget > 0 and self.cumulative_window_attempts <= budget:
            self.milestones["last_within_window_budget"] = {
                "round": 0,
                "target_window_budget": budget,
                "cumulative_window_attempts": self.cumulative_window_attempts,
                "cumulative_output_attempts": self.cumulative_output_attempts,
                "best_round": 0,
                "best_validation_mse": val["mean_mse"],
                "best_hash": digest(self.initial),
                "validation_at_round": val["mean_mse"],
                "checkpoint": "best-validation-within-window-budget.pt",
            }
            checkpoint(
                self.folder / "best-validation-within-window-budget.pt",
                self.model_at(self.center),
                {
                    "method": self.method,
                    "milestone": "last_within_window_budget",
                    "milestone_round": 0,
                    "selected_round": 0,
                    "validation_mse": val["mean_mse"],
                    "target_window_budget": budget,
                    "cumulative_window_attempts": self.cumulative_window_attempts,
                    "shared_pretrained_hash": digest(self.initial),
                },
            )

    def resume_from_attempt(self, prev_folder):
        return resume_study_from_attempt(self, prev_folder)

    def batch(self):
        needed = min(self.recipe.training_groups_per_round, self.client.info["trials"]["train"])
        values = []
        while len(values) < needed:
            if not len(self._order):
                self._order = np.random.default_rng(
                    np.random.SeedSequence([self.recipe.seed, 880, self._epoch])
                ).permutation(self.client.info["trials"]["train"])
                self._epoch += 1
            n = min(needed - len(values), len(self._order))
            values += self._order[:n].tolist()
            self._order = self._order[n:]
        return list(dict.fromkeys(values))

    def score(self, theta, split="train", trials=None, **kwargs):
        res = self.client.score(theta, split, trials, **kwargs)
        cost = res.get("cost", {}) if isinstance(res, dict) else {}
        self.__dict__["cumulative_window_attempts"] = self.__dict__.get(
            "cumulative_window_attempts", 0
        ) + int(cost.get("window_attempts", 0))
        self.__dict__["cumulative_output_attempts"] = self.__dict__.get(
            "cumulative_output_attempts", 0
        ) + int(cost.get("output_attempts", 0))
        return res

    def validate(self, k, row, force=False):
        row["round"] = k
        row["elapsed_seconds"] = getattr(self, "elapsed_offset", 0.0) + (
            time.monotonic() - self.started
        )
        budget = int(getattr(self.recipe, "target_window_budget", 0))
        cur_win_pre = self.__dict__.get("cumulative_window_attempts", 0)
        near_budget_boundary = (
            budget > 0
            and cur_win_pre <= budget
            and (cur_win_pre + 42 * 4 * self.recipe.training_groups_per_round) > budget
        )
        if (
            force
            or near_budget_boundary
            or k % self.recipe.validation_every_rounds == 0
            or k == self.recipe.fine_rounds
        ):
            val = self.score(self.center, "validation", None, label=f"round-{k}-validation")
            row["validation"] = val
            cur_win = self.__dict__.get("cumulative_window_attempts", 0)
            cur_out = self.__dict__.get("cumulative_output_attempts", 0)
            update_best_checkpoint(self, k, val, cur_win, row)
            if not val["failures"] and val["mean_mse"] < self.significant_best * (
                1 - self.recipe.plateau_relative
            ):
                self.significant_best = val["mean_mse"]
                self.bad_checks = 0
            else:
                self.bad_checks += 1
            self.stopped = self.bad_checks >= self.recipe.patience
            row["early_stopping"] = {
                "significant_best": self.significant_best,
                "bad_checks": self.bad_checks,
                "stop": self.stopped,
            }
            record_scheduled_milestones(self, k, val, budget, cur_win, cur_out)
        elif (
            budget > 0
            and cur_win_pre > budget
            and "first_exceeding_window_budget" not in self.__dict__.get("milestones", {})
        ):
            val = self.score(self.center, "validation", None, label=f"round-{k}-validation-budget")
            row["validation"] = val
            cur_win = self.__dict__.get("cumulative_window_attempts", 0)
            cur_out = self.__dict__.get("cumulative_output_attempts", 0)
            update_best_checkpoint(self, k, val, cur_win, row)
            milestones = self.__dict__.setdefault("milestones", {})
            record_budget_exceeding_milestone(self, k, val, budget, cur_win, cur_out, milestones)
        row["cumulative_window_attempts"] = self.__dict__.get("cumulative_window_attempts", 0)
        row["cumulative_output_attempts"] = self.__dict__.get("cumulative_output_attempts", 0)
        row["best_so_far"] = {
            "round": self.best[1],
            "validation_mse": self.best[0],
            "policy_hash": digest(self.best[2]),
            "checkpoint": f"checkpoints/best-round-{self.best[1]:04d}.pt",
        }
        self.history.append(sanitize_persisted_paths(row, self.folder))
        save(self.folder / "rounds" / f"round-{k:04d}.json", self.history[-1])
        checkpoint(
            self.folder / "latest.pt",
            self.model_at(self.center),
            {
                "method": self.method,
                "round": k,
                "shared_pretrained_hash": digest(self.initial),
            },
        )
        save(
            self.folder / "progress.json",
            {
                "status": "training",
                "method": self.method,
                "rounds_completed": len(self.history),
                "last_round": k,
                "cumulative_window_attempts": row["cumulative_window_attempts"],
                "milestones": self.__dict__.get("milestones", {}),
                "last_validation": next(
                    (
                        {"round": r["round"], "mse": r["validation"]["mean_mse"]}
                        for r in reversed(self.history)
                        if "validation" in r
                    ),
                    None,
                ),
                "round_records": "rounds/round-*.json",
                "planned_rounds": self.recipe.fine_rounds,
            },
        )
        if getattr(self.recipe, "ignore_early_stop", False):
            if budget > 0 and "first_exceeding_window_budget" in self.__dict__.get(
                "milestones", {}
            ):
                return True
            return False
        return self.stopped

    def model_at(self, theta):
        set_model(self.model, theta)
        return self.model

    def _build_cost_checkpoints(self, pre_finish_windows):
        return build_cost_checkpoints(self, pre_finish_windows)

    def finish(self):
        _, k, theta = self.best
        train_sel_win = int(getattr(self, "cumulative_window_attempts", 0))
        train_sel_out = int(getattr(self, "cumulative_output_attempts", 0))
        cost_checkpoints = self._build_cost_checkpoints(train_sel_win)
        checkpoint(
            self.folder / "selected.pt",
            self.model_at(theta),
            {
                "method": self.method,
                "selected_round": k,
                "shared_pretrained_hash": digest(self.initial),
            },
        )
        training_eval = (
            self.train
            if k == 0
            else self.score(theta, "train", None, label="selected-full-train-once")
        )
        total_win = int(getattr(self, "cumulative_window_attempts", 0))
        total_out = int(getattr(self, "cumulative_output_attempts", 0))
        result = {
            "status": "complete",
            "method": self.method,
            "start_hash": digest(self.initial),
            "selected_hash": digest(theta),
            "selected_round": k,
            "milestones": self.__dict__.get("milestones", {}),
            "cost_checkpoints": cost_checkpoints,
            "history": self.history,
            "evaluations": {"validation": self.best_score},
            "training": training_eval,
            "early_stopped": self.stopped,
            "seconds": getattr(self, "elapsed_offset", 0.0) + (time.monotonic() - self.started),
            "training_and_selection_window_attempts": train_sel_win,
            "training_and_selection_output_attempts": train_sel_out,
            "post_selection_window_attempts": total_win - train_sel_win,
            "post_selection_output_attempts": total_out - train_sel_out,
            "cumulative_window_attempts": total_win,
            "cumulative_output_attempts": total_out,
            "resume_provenance": getattr(self, "resume_provenance", None),
            "protocol": self.recipe.protocol,
            "test_id_deferred": self.recipe.defer_test_id,
            "actor_numeric": self.actor_numeric,
            "information": "shared initialization recorded in common.json; public F/H values and own G only",
        }
        result = sanitize_persisted_paths(result, self.folder)
        save(self.folder / "result.json", result)
        return result
