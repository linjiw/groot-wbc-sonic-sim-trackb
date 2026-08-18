"""LACE failure-geometry research primitives.

This package is deliberately independent of Isaac Lab so artifact contracts and
sampling safety can be validated on CPU before simulator integration.
"""

from gear_sonic.research.lace.atlas import build_atlas_manifest
from gear_sonic.research.lace.compute import (
    ComputeBudget,
    assert_headline_matched_training_budget,
    assert_matched_training_budget,
)
from gear_sonic.research.lace.interventions import solve_fixed_budget_intervention
from gear_sonic.research.lace.probes import ProbeThresholds, compute_episode_probe
from gear_sonic.research.lace.reference_feasibility import (
    ReferenceFeasibilityThresholds,
    compute_reference_feasibility_proxy,
)
from gear_sonic.research.lace.schedule import (
    build_rollout_schedule,
    derive_runtime_rng_seed,
    quantize_start_step,
    validate_rollout_schedule,
)
from gear_sonic.research.lace.schema import (
    ATLAS_SCHEMA_VERSION,
    SPLIT_SCHEMA_VERSION,
    canonical_sha256,
    validate_atlas_manifest,
    validate_split_manifest,
)
from gear_sonic.research.lace.signatures import (
    DEFAULT_MECHANISMS,
    build_factorized_signatures,
)
from gear_sonic.research.lace.sonic_lite import (
    SonicLiteParameterReport,
    audit_lite_s_profile,
)
from gear_sonic.research.lace.stability import (
    compare_split_half_signatures,
    cross_validated_difficulty_reconstruction,
)
from gear_sonic.research.lace.transfer_analysis import (
    analyze_transfer_predictiveness,
    bootstrap_clustered_loss_improvement,
    permute_whole_source_failure_rows,
)
from gear_sonic.research.lace.transfer_features import (
    build_signed_exposure_transfer_features,
)
from gear_sonic.research.lace.trust_region import (
    TrustRegionConstraints,
    apply_lace_tilt,
)

__all__ = [
    "ATLAS_SCHEMA_VERSION",
    "ComputeBudget",
    "DEFAULT_MECHANISMS",
    "ProbeThresholds",
    "ReferenceFeasibilityThresholds",
    "SPLIT_SCHEMA_VERSION",
    "SonicLiteParameterReport",
    "TrustRegionConstraints",
    "apply_lace_tilt",
    "analyze_transfer_predictiveness",
    "assert_headline_matched_training_budget",
    "assert_matched_training_budget",
    "audit_lite_s_profile",
    "build_atlas_manifest",
    "build_factorized_signatures",
    "build_rollout_schedule",
    "build_signed_exposure_transfer_features",
    "bootstrap_clustered_loss_improvement",
    "canonical_sha256",
    "compute_episode_probe",
    "compute_reference_feasibility_proxy",
    "compare_split_half_signatures",
    "cross_validated_difficulty_reconstruction",
    "derive_runtime_rng_seed",
    "quantize_start_step",
    "permute_whole_source_failure_rows",
    "solve_fixed_budget_intervention",
    "validate_atlas_manifest",
    "validate_rollout_schedule",
    "validate_split_manifest",
]
