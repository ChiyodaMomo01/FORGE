"""Interval-aligned public traces and same-reset output differences."""

import numpy as np


class IntervalTrace:
    """Only paired output differences have a known zero initial condition."""

    def __init__(self, raw):
        if raw.field_channels < 1:
            raise ValueError("at least one public output channel required")
        if raw.innovations is None:
            raise ValueError("recorded independent innovations required")
        self.raw = raw
        self.meta = raw.meta
        self.members = raw.members
        self.height = raw.height
        self.width = raw.width
        self.field_channels = int(raw.field_channels)
        self.action_channels = int(raw.action_channels)
        self.steps = raw.outputs
        self.starts = np.r_[0, raw.output_after[:-1] + 1].astype(int)
        if not np.array_equal(
            raw.intervals,
            np.repeat(np.arange(self.steps), np.diff(np.r_[self.starts, raw.steps])),
        ):
            raise ValueError("public callback/interval order mismatch")
        scale_vec = np.asarray(self.meta["field_std"], np.float32)
        if (
            scale_vec.shape != (self.field_channels,)
            or not np.isfinite(scale_vec).all()
            or np.any(scale_vec <= 0)
        ):
            raise ValueError("invalid train-only field scale")
        self.scale = float(scale_vec[0]) if self.field_channels == 1 else scale_vec
        self.scale_array = scale_vec.reshape(1, 1, self.field_channels, 1, 1)
        self.anchors = np.asarray(raw.observations[self.starts], np.float32)
        self.fields = np.asarray(raw.fields[:], np.float32) / self.scale_array
        self.forcing = []
        for start, after in zip(self.starts, raw.output_after, strict=True):
            values = raw.innovations[start : after + 1]
            if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                raise ValueError("probe must stay coherent throughout each public output interval")
            self.forcing.append(values[0])
        self.forcing = np.stack(self.forcing).astype(np.float32)
        if self.meta["sigma"] != 0 and not self.meta.get("coherent_interval"):
            raise ValueError("explicit coherent probe metadata required")

    def task_residual(self):
        reference = self.raw.task()["reference"].transpose(1, 0, 2, 3, 4) / self.scale_array
        return self.fields - reference

    def expand(self, correction):
        return correction[self.raw.intervals]


def paired(center, probe):
    if (
        center.fields.shape != probe.fields.shape
        or center.forcing.shape != probe.forcing.shape
        or center.anchors.shape != probe.anchors.shape
        or not np.array_equal(center.scale_array, probe.scale_array)
        or not np.array_equal(center.anchors[0], probe.anchors[0])
    ):
        raise ValueError("same-reset public shapes, initial observations and field scales required")
    if center.steps != probe.steps or not np.array_equal(
        center.raw.output_after, probe.raw.output_after
    ):
        raise ValueError("interval alignment required")
    if center.meta["policy_hash"] != probe.meta["policy_hash"]:
        raise ValueError("fixed-policy closed-loop probes required")
    if (
        center.meta["trial"] != probe.meta["trial"]
        or center.meta["sigma"] != 0
        or probe.meta["sigma"] == 0
    ):
        raise ValueError("same-window deterministic center and independent nonzero probe required")


def output_difference(center, probe):
    paired(center, probe)
    delta = probe.fields - center.fields
    # The reset identity gives delta_y_0=0; no callback channel is treated as y_0.
    return np.concatenate((np.zeros_like(delta[:1]), delta), axis=0)
