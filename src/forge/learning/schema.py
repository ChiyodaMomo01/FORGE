"""Public window contract and predeclared training recipe; no physics imports."""

from __future__ import annotations

from ..core.contracts import (
    ACTIVE_SPLITS,
    CH_RECIPE,
    GS_RECIPE,
    METHODS,
    MHD_RECIPE,
    NS_RECIPE,
    PROTOCOLS,
    RECIPE,
    RECIPE_BY_CASE,
    Recipe,
    apply_validated_recipe_overrides,
    epoch_batches,
    load_recipe_profile,
    trial_map,
    validate_profile_data,
)

__all__ = [
    "ACTIVE_SPLITS",
    "CH_RECIPE",
    "GS_RECIPE",
    "METHODS",
    "MHD_RECIPE",
    "NS_RECIPE",
    "PROTOCOLS",
    "RECIPE",
    "RECIPE_BY_CASE",
    "Recipe",
    "apply_validated_recipe_overrides",
    "epoch_batches",
    "load_recipe_profile",
    "trial_map",
    "validate_profile_data",
]
