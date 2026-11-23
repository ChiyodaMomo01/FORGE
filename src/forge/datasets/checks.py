"""Array contract validation and physical invariant checks for 2D benchmarks."""

import numpy as np

from ..config import COMPONENT_LABEL_CHANNELS

FIELD_CHANNELS = {
    "gray_scott": 2,
    "cahn_hilliard": 1,
    "kolmogorov_ns": 1,
    "mhd2d": 2,
}


def archive_checks(archive, task, storage_dtype):
    """Validate stored archive arrays against the task contract."""
    issues = []
    n, frames = task["lr_grid"], task["frames"]
    fields = FIELD_CHANNELS[task["case"]]
    expected = {"fields": (frames + 1, fields, n, n)}
    if task["split"] == "train" and task["case"] in COMPONENT_LABEL_CHANNELS:
        expected["actions"] = (frames, COMPONENT_LABEL_CHANNELS[task["case"]], n, n)
    elif task["case"] not in COMPONENT_LABEL_CHANNELS and "actions" in archive:
        issues.append("undefined component labels")
    elif "actions" in archive:
        issues.append("held-out labels")
    if "times" not in archive or not np.array_equal(
        archive["times"], np.arange(frames + 1) * task["sample_dt"]
    ):
        issues.append("wrong or missing saved times")
    for name, shape in expected.items():
        if name not in archive:
            issues.append("missing " + name)
            continue
        array = archive[name]
        if array.shape != shape:
            issues.append("wrong stored " + name + " shape")
        if array.dtype != np.dtype(storage_dtype):
            issues.append("wrong stored " + name + " dtype")
        if not np.isfinite(array).all():
            issues.append("nonfinite stored " + name)
    return issues


def physical_checks(name, rows):
    """Verify physical invariants and numerical stability bounds on HR diagnostics."""
    issues = []
    metrics = {}

    def series(key):
        return np.asarray([r[key] for r in rows], dtype=np.float64)

    if not rows:
        return {"issues": ["no HR diagnostics"], "metrics": {}}

    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return bool(np.isfinite(value)) if isinstance(value, (int, float)) else True

    if not finite(rows):
        return {"issues": ["nonfinite HR diagnostics"], "metrics": {}}
    if name == "cahn_hilliard":
        mean, energy = series("mean"), series("energy")
        metrics["max_mass_drift"] = float(abs(mean - mean[0]).max())
        metrics["max_energy_increase"] = float(
            np.maximum(np.diff(energy, axis=0), 0).max(initial=0)
        )
        if metrics["max_mass_drift"] > 1e-9:
            issues.append("CH mass drift > 1e-9")
        if metrics["max_energy_increase"] > 1e-10:
            issues.append("CH free energy increase > 1e-10")
    if name == "gray_scott":
        metrics["minimum_concentration"] = float(series("minimum").min())
        metrics["active_fraction_range"] = [
            float(series("active_fraction_v_gt_0.1").min()),
            float(series("active_fraction_v_gt_0.1").max()),
        ]
        if metrics["minimum_concentration"] < -1e-10:
            issues.append("GS negative concentration")
    if name == "kolmogorov_ns":
        metrics["max_advective_cfl"] = float(series("max_advective_cfl").max())
        metrics["max_mean_abs"] = float(series("mean_abs").max())
        metrics["max_divergence_l2"] = float(series("divergence_l2").max())
        metrics["kinetic_energy_range"] = [
            float(series("kinetic_energy").min()),
            float(series("kinetic_energy").max()),
        ]
        if metrics["max_advective_cfl"] > 1.0:
            issues.append("NS advective CFL > 1")
        if metrics["max_mean_abs"] > 1e-10:
            issues.append("NS vorticity mean drift > 1e-10")
        if metrics["max_divergence_l2"] > 1e-10:
            issues.append("NS velocity divergence > 1e-10")
        if metrics["kinetic_energy_range"][0] <= 0.0:
            issues.append("NS nonpositive kinetic energy")
    if name == "mhd2d":
        metrics["max_advective_cfl"] = float(series("max_advective_cfl").max())
        metrics["max_alfven_cfl"] = float(series("max_alfven_cfl").max())
        metrics["max_mean_abs"] = float(series("mean_abs").max())
        metrics["max_divergence_u_l2"] = float(series("divergence_u_l2").max())
        metrics["max_divergence_b_l2"] = float(series("divergence_b_l2").max())
        metrics["kinetic_energy_range"] = [
            float(series("kinetic_energy").min()),
            float(series("kinetic_energy").max()),
        ]
        metrics["magnetic_energy_range"] = [
            float(series("magnetic_energy").min()),
            float(series("magnetic_energy").max()),
        ]
        if metrics["max_advective_cfl"] > 1.0:
            issues.append("MHD advective CFL > 1")
        if metrics["max_mean_abs"] > 1e-10:
            issues.append("MHD field/potential mean drift > 1e-10")
        if metrics["max_divergence_u_l2"] > 1e-10:
            issues.append("MHD velocity divergence > 1e-10")
        if metrics["max_divergence_b_l2"] > 1e-10:
            issues.append("MHD magnetic divergence > 1e-10")
        if metrics["kinetic_energy_range"][0] <= 0.0:
            issues.append("MHD nonpositive kinetic energy")
        if metrics["magnetic_energy_range"][0] <= 1e-6:
            issues.append("MHD magnetic energy collapsed <= 1e-6")
    return {"issues": issues, "metrics": metrics}
