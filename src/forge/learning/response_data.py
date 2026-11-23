"""Consecutive public-array traces for response fitting, separate HR task labels.

No dataset, state reconstruction, native solver, configuration or operators are
available here. The terminal next observation is explicitly missing, not zero.
"""

import json

import numpy as np

from ..core.io import load_array, resolve_archived_path
from .events import RecordedArray


def header(path):
    value = load_array(path, mmap_mode="r", allow_pickle=False)
    try:
        return value.shape, value.dtype
    finally:
        getattr(value, "_mmap", None) is not None and value._mmap.close()


class ResponseTrace:
    def __init__(self, path):
        self.path = resolve_archived_path(path)
        self.folder = self.path.parent
        self.meta = json.loads(self.path.read_text())
        if self.meta.get("schema") != "public-response-v1" or self.meta.get("split") != "train":
            raise ValueError("training public-response trace required")
        if not self.meta.get("complete"):
            raise ValueError("failed/incomplete traces are evidence, not complete response targets")
        shape, dtype = header(self.folder / "observations.npy")
        if len(shape) != 5 or shape[1] != 4 or shape[2] not in (2, 4, 6, 8) or dtype != np.float32:
            raise ValueError("authorized joint public observation shape/dtype mismatch")
        self.steps, self.members, self.channels, self.height, self.width = shape
        action_shape, action_dtype = header(self.folder / "actions.npy")
        field_shape, field_dtype = header(self.folder / "fields.npy")
        if len(action_shape) != 5 or len(field_shape) != 5:
            raise ValueError("invalid public response channel axes")
        self.action_channels = action_shape[2]
        self.field_channels = field_shape[2]
        case_by_channels = {
            (2, 1, 1): "cahn_hilliard",
            (4, 1, 2): "gray_scott",
            (6, 2, 1): "kolmogorov_ns",
            (8, 3, 2): "mhd2d",
        }
        if (
            self.channels,
            self.action_channels,
            self.field_channels,
        ) not in case_by_channels:
            raise ValueError("unauthorized public channel contract")
        self.case_name = case_by_channels[
            (self.channels, self.action_channels, self.field_channels)
        ]
        if self.meta.get("case_name", self.case_name) != self.case_name:
            raise ValueError("public case/channel disagreement")
        if self.steps < 2 or self.steps != self.meta["callback_count"]:
            raise ValueError("invalid callback count")
        self.output_indices = load_array(self.folder / "output_indices.npy", allow_pickle=False)
        self.output_after = load_array(
            self.folder / "output_after_sequence.npy", allow_pickle=False
        )
        self.intervals = load_array(self.folder / "intervals.npy", allow_pickle=False)
        self.outputs = len(self.output_after)
        if (
            not np.array_equal(self.output_indices, np.arange(self.outputs))
            or self.outputs != self.meta["output_count"]
        ):
            raise ValueError("missing or reordered public outputs")
        if (
            np.any(np.diff(self.output_after) <= 0)
            or np.any(self.output_after < 0)
            or np.any(self.output_after >= self.steps)
        ):
            raise ValueError("invalid output-action pairing")
        if self.intervals.shape != (self.steps,) or np.any(np.diff(self.intervals) < 0):
            raise ValueError("invalid public interval sequence")
        for index, after in enumerate(self.output_after):
            if self.intervals[after] != index:
                raise ValueError("public output/callback interval disagreement")
        for name, expected in [
            ("actions", (self.steps, 4, self.action_channels, self.height, self.width)),
            ("fields", (self.outputs, 4, self.field_channels, self.height, self.width)),
        ]:
            s, d = header(self.folder / (name + ".npy"))
            if s != expected or d != np.float32:
                raise ValueError("invalid public " + name + " array")
        self.observations = RecordedArray(self.folder / "observations.npy")
        self.actions = RecordedArray(self.folder / "actions.npy")
        self.innovations = (
            RecordedArray(self.folder / "innovations.npy")
            if (self.folder / "innovations.npy").exists()
            or (self.folder / "innovations.npy.gz").exists()
            else None
        )
        if self.innovations is not None:
            shape, dtype = header(self.folder / "innovations.npy")
            if shape != action_shape or dtype != np.float32:
                raise ValueError("invalid independent action innovation array")
        self.fields = RecordedArray(self.folder / "fields.npy")

    def task(self):
        """HR reference access is explicit and separate from E-fitting batches."""
        reference = load_array(self.folder / "reference.npy", allow_pickle=False)
        if (
            reference.shape != (4, self.outputs, self.field_channels, self.height, self.width)
            or not np.isfinite(reference).all()
        ):
            raise ValueError("invalid public HR task reference")
        scale = np.asarray(self.meta["field_std"], np.float64)
        if (
            scale.shape != (self.field_channels,)
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("invalid train-only field scale")
        return {
            "initial_observation": self.observations[0],
            "reference": reference,
            "output_after_sequence": self.output_after.copy(),
            "field_std": scale,
        }
