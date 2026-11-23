"""Two-stage FORGE 3D POM training algorithms, causal response operators, and ablations."""

from .response_operator import (
    CausalBlockResponseOperator3D,
    CausalObservationResponseOperator3D,
    apply_obs_step_differentiable,
    apply_response_trajectory_differentiable,
)
from .stage1 import Stage1DriftCompensator, run_stage1_subspace_step
from .stage2 import (
    build_projected_action_probe_direction,
    measure_action_jacobian_ranks,
    run_stage2_response_step,
    solve_replay_head_ridge,
)
from .trainer import (
    VALID_ABLATION_VARIANTS,
    run_forge_main_4h,
    run_stage2_forge_main_4h,
)

__all__ = [
    "CausalBlockResponseOperator3D",
    "CausalObservationResponseOperator3D",
    "Stage1DriftCompensator",
    "VALID_ABLATION_VARIANTS",
    "apply_obs_step_differentiable",
    "apply_response_trajectory_differentiable",
    "build_projected_action_probe_direction",
    "measure_action_jacobian_ranks",
    "run_forge_main_4h",
    "run_stage1_subspace_step",
    "run_stage2_forge_main_4h",
    "run_stage2_response_step",
    "solve_replay_head_ridge",
]
