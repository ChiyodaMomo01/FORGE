"""Reference trajectory dataset indexed strictly by physical model time."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import (
    CANONICAL_CORE_NAME,
    CANONICAL_NUMERICS_VERSION,
    normalize_generation_variant,
    validate_stored_core_identity,
)
from .core_spec import core_spec

CORE = CANONICAL_CORE_NAME
NUMERICS_VERSION = CANONICAL_NUMERICS_VERSION
COARSE_DEPLOYMENT_DTI_SECONDS = 600.0
COARSE_DEPLOYMENT_DTE_SECONDS = 20.0
COARSE_DEPLOYMENT_ISPLIT = 30
# Accumulated fine-step clock drift reaches ~1e-7 day near production day 9000.
NOMINAL_CLOCK_TOL_DAYS = 2e-7
LABEL_FIELDS = ("u", "ub", "v", "vb", "t", "s", "el", "ua", "uab", "va", "vab")
INITIAL_3D_FIELDS = (
    "u",
    "ub",
    "v",
    "vb",
    "t",
    "tb",
    "s",
    "sb",
    "q2",
    "q2b",
    "q2l",
    "q2lb",
    "km",
    "kh",
    "kq",
    "aam",
    "rho",
    "w",
)
INITIAL_2D_FIELDS = (
    "el",
    "elb",
    "et",
    "etb",
    "etf",
    "ua",
    "uab",
    "va",
    "vab",
    "utf",
    "utb",
    "vtf",
    "vtb",
    "adx2d",
    "ady2d",
    "drx2d",
    "dry2d",
    "vfluxf",
    "vfluxb",
    "wubot",
    "wvbot",
    "egb",
    "uaf",
    "vaf",
    "elf",
)


def _is_strict_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _require_finite_real(value: Any, *, key: str, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{context} numerical setting {key}={value!r} must be a real number")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{context} numerical setting {key}={value!r} must be finite")
    return fval


def _resolve_downsample_factor(downsampling: Any) -> int:
    if not isinstance(downsampling, dict):
        raise ValueError(
            f"Dataset manifest 'downsampling' must be a JSON object, got {downsampling!r}"
        )
    if "enabled" in downsampling:
        enabled = downsampling["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError(
                f"Dataset manifest downsampling.enabled must be a boolean, got {enabled!r}"
            )
    else:
        enabled = False

    if not enabled:
        if "factor" in downsampling:
            raw_factor = downsampling["factor"]
            if not _is_strict_int(raw_factor) or int(raw_factor) != 1:
                raise ValueError(f"Disabled downsampling cannot specify factor={raw_factor!r} != 1")
        return 1

    if "factor" not in downsampling:
        raise ValueError("Enabled dataset downsampling must specify integer 'factor'")
    raw_factor = downsampling["factor"]
    if not _is_strict_int(raw_factor) or int(raw_factor) < 1:
        raise ValueError(f"Invalid downsampling factor in dataset manifest: {raw_factor!r}")
    factor = int(raw_factor)
    if int(COARSE_DEPLOYMENT_DTI_SECONDS) % factor != 0 or (
        int(COARSE_DEPLOYMENT_DTE_SECONDS) % factor != 0
    ):
        raise ValueError(
            f"Downsampling factor={factor} does not divide coarse deployment steps "
            f"({COARSE_DEPLOYMENT_DTI_SECONDS}s, {COARSE_DEPLOYMENT_DTE_SECONDS}s)"
        )
    return factor


def _validate_integration_dict(
    numerics: Any,
    context: str,
    *,
    downsample_factor: int = 1,
    require_step_fields: bool = False,
) -> tuple[float, float, int]:
    """Validate HR generation and LR deployment step/filter parameters with NaN/type guards."""
    if not isinstance(numerics, dict):
        raise ValueError(f"{context} numerical settings must be a JSON object, got {numerics!r}")
    if not _is_strict_int(downsample_factor) or int(downsample_factor) < 1:
        raise ValueError(f"Invalid downsample_factor={downsample_factor!r}")

    expected_hr_dti = COARSE_DEPLOYMENT_DTI_SECONDS / float(downsample_factor)
    expected_hr_dte = COARSE_DEPLOYMENT_DTE_SECONDS / float(downsample_factor)

    has_dti = "dti_seconds" in numerics or "dti" in numerics
    has_dte = "dte_seconds" in numerics or "dte" in numerics
    has_isplit = "isplit" in numerics
    if require_step_fields and not (has_dti and has_dte and has_isplit):
        raise ValueError(
            f"{context} missing required step settings (dti/dti_seconds, dte/dte_seconds, isplit)"
        )

    for key, expected in (
        ("dti_seconds", expected_hr_dti),
        ("dti", expected_hr_dti),
        ("dte_seconds", expected_hr_dte),
        ("dte", expected_hr_dte),
        ("smoth", 0.15),
        ("alpha", 0.333),
        ("horcon", 0.05),
    ):
        if key in numerics:
            fval = _require_finite_real(numerics[key], key=key, context=context)
            if not (abs(fval - expected) <= 1e-9):
                raise ValueError(
                    f"{context} numerical setting {key}={numerics[key]!r} != {expected}"
                )

    if "isplit" in numerics:
        raw_isplit = numerics["isplit"]
        if not _is_strict_int(raw_isplit) or int(raw_isplit) != COARSE_DEPLOYMENT_ISPLIT:
            raise ValueError(
                f"{context} numerical setting isplit={raw_isplit!r} != {COARSE_DEPLOYMENT_ISPLIT}"
            )

    eff_hr_dti = (
        float(numerics["dti_seconds"])
        if "dti_seconds" in numerics
        else float(numerics["dti"])
        if "dti" in numerics
        else expected_hr_dti
    )
    eff_hr_dte = (
        float(numerics["dte_seconds"])
        if "dte_seconds" in numerics
        else float(numerics["dte"])
        if "dte" in numerics
        else expected_hr_dte
    )
    if not (abs(eff_hr_dti - eff_hr_dte * float(COARSE_DEPLOYMENT_ISPLIT)) <= 1e-9):
        raise ValueError(
            f"{context} mode-split step mismatch: dti={eff_hr_dti} != dte={eff_hr_dte} * {COARSE_DEPLOYMENT_ISPLIT}"
        )
    if not (
        abs(eff_hr_dti * float(downsample_factor) - COARSE_DEPLOYMENT_DTI_SECONDS) <= 1e-9
        and abs(eff_hr_dte * float(downsample_factor) - COARSE_DEPLOYMENT_DTE_SECONDS) <= 1e-9
    ):
        raise ValueError(
            f"{context} HR step ({eff_hr_dti}, {eff_hr_dte}) * factor={downsample_factor} "
            f"does not match coarse deployment step ({COARSE_DEPLOYMENT_DTI_SECONDS}, {COARSE_DEPLOYMENT_DTE_SECONDS})"
        )
    return eff_hr_dti, eff_hr_dte, COARSE_DEPLOYMENT_ISPLIT


class FixedCoreDataset:
    """Read-only coarse-grained reference ensemble reader with strict numerical verification."""

    def __init__(
        self,
        root: str | Path,
        frequency_seconds: int = 86400,
        core_name: str = CORE,
        expected_grid: tuple[int, int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.spec = core_spec(core_name)
        if not _is_strict_int(frequency_seconds) or int(frequency_seconds) not in (3600, 86400):
            raise ValueError("Correction interval must be integer 3600 (1 h) or 86400 (1 day)")
        self.frequency_seconds = int(frequency_seconds)

        manifest_path = self.root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(self.manifest, dict):
            raise ValueError("Dataset manifest.json must be a JSON object")

        canonical_core, canonical_numerics = validate_stored_core_identity(
            self.manifest.get("generation_core_requested"),
            self.manifest.get("generation_core_numerics_version"),
        )
        if canonical_core != self.spec.name or canonical_numerics != self.spec.numerics_version:
            raise ValueError("Dataset does not match the registered dynamical core")

        shapiro_flag = self.manifest.get("implicit_spatial_shapiro_requested")
        if not isinstance(shapiro_flag, str) or shapiro_flag.lower() not in (
            "0",
            "false",
            "off",
            "no",
        ):
            raise ValueError("Semi-implicit 3D POM dataset requires spatial Shapiro OFF")

        downsampling = self.manifest.get("downsampling", {})
        self.downsample_factor = _resolve_downsample_factor(downsampling)

        raw_grid = downsampling.get("coarse_grid", (90, 82))
        if not isinstance(raw_grid, (list, tuple)) or len(raw_grid) not in (2, 3):
            raise ValueError(f"Invalid coarse_grid in dataset manifest: {raw_grid!r}")
        if any(not _is_strict_int(dim) for dim in raw_grid):
            raise ValueError(f"coarse_grid entries must be integers, got {raw_grid!r}")
        if len(raw_grid) == 3 and int(raw_grid[2]) != 16:
            raise ValueError(f"Expected 16 vertical sigma levels, got {raw_grid!r}")
        im_g, jm_g = int(raw_grid[0]), int(raw_grid[1])
        if im_g <= 0 or jm_g <= 0:
            raise ValueError(f"Invalid coarse_grid dimensions: {raw_grid!r}")
        self.grid: tuple[int, int, int] = (im_g, jm_g, 16)

        if "fine_grid" in downsampling:
            fine_grid = downsampling["fine_grid"]
            if (
                not isinstance(fine_grid, (list, tuple))
                or len(fine_grid) not in (2, 3)
                or any(not _is_strict_int(dim) for dim in fine_grid)
            ):
                raise ValueError(f"Invalid fine_grid in dataset manifest: {fine_grid!r}")
            expected_fine = (
                self.downsample_factor * (im_g - 2) + 2,
                self.downsample_factor * (jm_g - 2) + 2,
            )
            if (int(fine_grid[0]), int(fine_grid[1])) != expected_fine:
                raise ValueError(
                    f"Dataset fine_grid={fine_grid!r} inconsistent with coarse_grid={(im_g, jm_g)} "
                    f"and factor={self.downsample_factor} (expected {expected_fine})"
                )

        if expected_grid is not None and self.grid != tuple(expected_grid):
            raise ValueError(
                f"Dataset coarse grid {self.grid} does not match expected {tuple(expected_grid)}"
            )

        self._manifest_has_schema_v = "schema_version" in self.manifest
        if (
            self._manifest_has_schema_v
            or self.downsample_factor > 1
            or "integration" in self.manifest
        ):
            if "integration" not in self.manifest:
                raise ValueError("Dataset manifest missing required 'integration' settings")
            self._expected_hr_steps = _validate_integration_dict(
                self.manifest["integration"],
                "Dataset manifest",
                downsample_factor=self.downsample_factor,
                require_step_fields=True,
            )
        else:
            self._expected_hr_steps = (
                COARSE_DEPLOYMENT_DTI_SECONDS,
                COARSE_DEPLOYMENT_DTE_SECONDS,
                COARSE_DEPLOYMENT_ISPLIT,
            )

        raw_algo = downsampling.get("algorithm")
        if raw_algo is not None and (not isinstance(raw_algo, str) or not raw_algo.strip()):
            raise ValueError(f"Invalid downsampling.algorithm: {raw_algo!r}")

        dataset_name = self.root.name or self.root.resolve().name
        self.data_identity: dict[str, Any] = {
            "dataset_name": dataset_name,
            "manifest_sha256": manifest_sha256,
            "core": canonical_core,
            "numerics_version": canonical_numerics,
            "grid": list(self.grid),
            "generation_variant": normalize_generation_variant(
                self.manifest.get("generation_variant")
            ),
            "downsampling_algorithm": raw_algo,
        }
        self._verified_members: set[int] = set()

    def _member(self, member: int) -> Path:
        if not _is_strict_int(member) or not 0 <= int(member) < 8:
            raise ValueError("Member must be an integer in [0, 7]")
        member_idx = int(member)
        member_rel = f"member_{member_idx}_gpu{member_idx}"
        base = self.root / member_rel
        if member_idx not in self._verified_members:
            provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
            if not isinstance(provenance, dict):
                raise ValueError(f"Invalid provenance.json in {member_rel}")
            prov_core, prov_numerics = validate_stored_core_identity(
                provenance.get("core_requested"),
                provenance.get("core_numerics_version"),
            )
            raw_prov_member = provenance.get("member")
            if (
                not _is_strict_int(raw_prov_member)
                or int(raw_prov_member) != member_idx
                or prov_core != self.spec.name
                or prov_numerics != self.spec.numerics_version
            ):
                raise ValueError(f"Member core provenance mismatch: {member_rel}")
            if "resolved_numerics" not in provenance:
                raise ValueError(f"Missing resolved_numerics in {member_rel}")
            numerics = provenance["resolved_numerics"]
            if not isinstance(numerics, dict):
                raise ValueError(f"Invalid resolved_numerics in {member_rel}")
            if numerics.get("spatial_shapiro_enabled") is not False:
                raise ValueError(f"Member data requires spatial Shapiro OFF: {member_rel}")
            for gamma_key in (
                "barotropic_shapiro_gamma",
                "tracer_shapiro_gamma",
                "velocity_shapiro_gamma",
            ):
                if gamma_key not in numerics:
                    raise ValueError(f"Missing {gamma_key} in {member_rel} provenance")
                gval = _require_finite_real(
                    numerics[gamma_key], key=gamma_key, context=f"Member {member_rel}"
                )
                if gval != 0.0:
                    raise ValueError(f"Member data requires {gamma_key}=0.0: {member_rel}")

            require_member_steps = (
                "schema_version" in provenance
                or self._manifest_has_schema_v
                or self.downsample_factor > 1
            )
            member_steps = _validate_integration_dict(
                numerics,
                f"Member {member_rel}",
                downsample_factor=self.downsample_factor,
                require_step_fields=require_member_steps,
            )
            if (
                abs(member_steps[0] - self._expected_hr_steps[0]) > 1e-9
                or abs(member_steps[1] - self._expected_hr_steps[1]) > 1e-9
                or member_steps[2] != self._expected_hr_steps[2]
            ):
                raise ValueError(
                    f"Member {member_rel} step settings {member_steps} conflict with manifest {self._expected_hr_steps}"
                )

            algorithm = self.data_identity["downsampling_algorithm"]
            if algorithm is not None and provenance.get("downsampling_algorithm") != algorithm:
                raise ValueError(f"Member downsampling differs from dataset manifest: {member_rel}")
            self._verified_members.add(member_idx)
        return base

    def initial_path(self, member: int, model_day: int) -> Path:
        if not _is_strict_int(model_day) or not 1 <= int(model_day) <= 9000:
            raise ValueError("Initial model day must be an integer in [1, 9000]")
        return (
            self._member(member)
            / "checkpoints_lowres_consistent"
            / f"checkpoint_Day{int(model_day) - 1:04d}.npz"
        )

    def label_path(self, member: int, model_day: int, lead_seconds: int) -> Path:
        if (
            not _is_strict_int(model_day)
            or not _is_strict_int(lead_seconds)
            or int(lead_seconds) < 0
        ):
            raise ValueError(
                "Labels require integer model days and nonnegative integer lead seconds"
            )
        whole_days, within_day = divmod(int(lead_seconds), 86400)
        day = int(model_day) + whole_days
        if within_day == 0:
            return self.initial_path(member, day)
        if within_day > 14400 or within_day % 1200:
            raise ValueError("Intraday labels exist only every 20 min in the first 4 h")
        if not 0 <= day <= 8999:
            raise ValueError("Reference day outside registered coverage")
        segment = within_day // 1200 - 1
        return (
            self._member(member)
            / "reference_data_lowres_consistent"
            / f"ref_Day{day:04d}_seg{segment:02d}.npz"
        )

    @staticmethod
    def _read(
        path: Path,
        expected_time: float,
        map_location: str | torch.device,
        fields: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        with np.load(path, allow_pickle=False) as archive:
            if "time" not in archive:
                raise ValueError(f"Missing 'time' field in {path.name}")
            raw_time = float(archive["time"])
            if not math.isfinite(raw_time) or not (
                abs(raw_time - expected_time) <= NOMINAL_CLOCK_TOL_DAYS
            ):
                raise ValueError(
                    f"Reference clock mismatch in {path.name}; expected {expected_time}"
                )
            keys = archive.files if fields is None else (*fields, "time")
            result = {
                key: torch.as_tensor(
                    archive[key].copy(),
                    device=map_location,
                    dtype=torch.float64 if key == "time" else torch.float32,
                )
                for key in keys
            }
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise ValueError(f"Nonfinite reference field in {path.name}")
        return result

    def _validate_shapes(
        self, state: dict[str, torch.Tensor], *, initial: bool
    ) -> dict[str, torch.Tensor]:
        fields_3d = INITIAL_3D_FIELDS if initial else LABEL_FIELDS[:6]
        fields_2d = INITIAL_2D_FIELDS if initial else LABEL_FIELDS[6:]
        scalars = ("time", "iint") if initial else ("time",)
        required = (*fields_3d, *fields_2d, *scalars)
        missing = set(required) - set(state)
        if missing:
            raise ValueError(f"Missing required physical fields: {sorted(missing)}")
        for field in required:
            expected = (
                self.grid if field in fields_3d else self.grid[:2] if field in fields_2d else ()
            )
            if tuple(state[field].shape) != expected:
                raise ValueError(
                    f"Physical field {field} shape {tuple(state[field].shape)} != {expected}"
                )
        return state

    def load_initial(
        self,
        member: int,
        model_day: int,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        state = self._read(self.initial_path(member, model_day), float(model_day), map_location)
        return self._validate_shapes(state, initial=True)

    def load_target(
        self,
        member: int,
        model_day: int,
        lead_seconds: int,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        state = self._read(
            self.label_path(member, model_day, lead_seconds),
            model_day + lead_seconds / 86400.0,
            map_location,
            LABEL_FIELDS,
        )
        return self._validate_shapes(state, initial=False)

    target = load_target
