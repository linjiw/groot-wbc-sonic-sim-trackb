"""SONIC 64D latent parity between Python training/eval and the C++ deploy path.

Motivation
----------

``action.motion_token`` is the only feature in the synthetic dataset whose
meaning is defined by a *different codebase* than the one that writes it. The
C++ runtime feeds a 64-value ``token_state`` observation into the control
policy; the Python recorder writes ``env._full_latent``. If those two are not
the same representation, every exported episode teaches a VLA to emit tokens
the deployed decoder was never trained to accept, and nothing else in the
pipeline would notice.

What the two paths actually compute
-----------------------------------

Python (:mod:`gear_sonic.trl.modules.universal_token_modules`)::

    latent            = encoder(tokenizer_obs)               # (N, num_tokens, token_dim)
    encoded_tokens    = quantizer(latent)                    # FSQ, post-quantization
    all_tokens        = assemble_all_tokens(encoded_tokens)  # (B, S, num_tokens, token_dim)
    all_tokens       += residual                             # iff latent_residual_mode
                                                             #     == "post_quantization"
    _last_full_latent_flat = all_tokens.view(*all_tokens.shape[:-2], -1)

C++ deploy encoder ONNX (:func:`gear_sonic.utils.inference_helpers.
export_universal_token_encoders_as_onnx`)::

    encoded_tokens, _ = module.encode(encoder_name, ...)      # FSQ, post-quantization
    selected_tokens   = ....flatten(start_dim=-2)             # (B, num_tokens*token_dim)

Both therefore occupy the **same slot** (the decoder ``token_flattened``
input), use the **same post-quantization representation**, and use the **same
row-major (num_tokens, token_dim) flatten order**. That much is parity.

The one asymmetry that this module exists to measure
-----------------------------------------------------

The C++ runtime contains no residual path at all: ``token_state`` is either the
raw encoder ONNX output or an external token vector supplied over ZMQ/ROS2. The
Python wrapper, by contrast, defaults ``action_mode`` to ``"residual"`` even
when ``use_latent_residual`` is false, and adds the policy's first 64 meta-action
values on top of the quantized tokens.

Post-quantization FSQ output lives on an exact rational lattice, so the two
cases are *numerically distinguishable* from the recorded array alone:

* on-lattice  -> byte-comparable to a local-encoder ``token_state``;
* off-lattice -> a residual-carrying decoder input. Still the correct target for
  a VLA that drives the decoder directly, but **not** interchangeable with
  encoder-sourced tokens recorded from a real teleop session.

Mixing both conventions under one feature name inside a single dataset is the
failure this gate is designed to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "DEPLOY_RELEASE_OBSERVATION_CONFIG",
    "LatentParityReport",
    "SONIC_RELEASE_TOKEN_CONTRACT",
    "TokenContract",
    "check_latent_parity",
    "flatten_tokens",
    "fsq_level_code_values",
    "read_deploy_encoder_dimension",
    "unflatten_tokens",
]


# Path is relative to the repository root; resolved by the caller or by
# :func:`read_deploy_encoder_dimension` when given an explicit path.
DEPLOY_RELEASE_OBSERVATION_CONFIG = "gear_sonic_deploy/policy/release/observation_config.yaml"


@dataclass(frozen=True)
class TokenContract:
    """The FSQ token geometry shared by the Python and C++ latent slots.

    Mirrors ``max_num_tokens`` / ``num_fsq_levels`` / ``fsq_level_list`` from an
    ``actor_critic/universal_token/*.yaml`` config. ``fsq_level_list`` is stored
    per token dimension so that a future non-uniform codebook stays correct.
    """

    max_num_tokens: int
    fsq_level_list: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.max_num_tokens < 1:
            raise ValueError("max_num_tokens must be positive")
        if not self.fsq_level_list:
            raise ValueError("fsq_level_list must be non-empty")
        for level in self.fsq_level_list:
            if not isinstance(level, int) or isinstance(level, bool) or level < 2:
                raise ValueError(f"FSQ level must be an integer >= 2; got {level!r}")

    @property
    def token_dim(self) -> int:
        """Dimensionality of a single token (equals ``num_fsq_levels``)."""
        return len(self.fsq_level_list)

    @property
    def total_dim(self) -> int:
        """Flattened latent width, i.e. the deployment ``token_state`` size."""
        return self.max_num_tokens * self.token_dim

    def flat_levels(self) -> np.ndarray:
        """Per-element FSQ level for the flattened latent, in row-major order."""
        return np.tile(np.asarray(self.fsq_level_list, dtype=np.int64), self.max_num_tokens)


#: The released SONIC checkpoint: ``actor_critic/universal_token/all_mlp_v1``
#: declares ``num_fsq_levels: 32``, ``fsq_level_list: 32``, ``max_num_tokens: 2``
#: which yields the 64-value ``token_state`` that
#: ``gear_sonic_deploy/policy/release/observation_config.yaml`` declares.
SONIC_RELEASE_TOKEN_CONTRACT = TokenContract(
    max_num_tokens=2,
    fsq_level_list=(32,) * 32,
)


def fsq_level_code_values(level: int) -> np.ndarray:
    """Return every scalar value ``vector_quantize_pytorch.FSQ`` can emit.

    Derived from ``FSQ.bound``::

        half_l  = (L - 1) * (1 + eps) / 2
        offset  = 0.5 if L is even else 0.0
        bounded = tanh(z + atanh(offset / half_l)) * half_l - offset
        code    = round(bounded) / (L // 2)

    ``tanh`` is a strict bijection onto ``(-1, 1)``, so ``bounded`` is confined
    to the open interval ``(-half_l - offset, half_l - offset)``. With the
    upstream ``eps = 1e-3`` the reachable rounded integers are exactly ``L``
    consecutive values, symmetric for odd ``L`` and biased one step negative for
    even ``L``.
    """
    if not isinstance(level, (int, np.integer)) or isinstance(level, bool) or level < 2:
        raise ValueError(f"FSQ level must be an integer >= 2; got {level!r}")
    level = int(level)
    half_width = level // 2
    if level % 2 == 0:
        codes = np.arange(-half_width, half_width, dtype=np.float64)
    else:
        codes = np.arange(-half_width, half_width + 1, dtype=np.float64)
    return codes / float(half_width)


def flatten_tokens(tokens: np.ndarray, contract: TokenContract) -> np.ndarray:
    """Flatten ``(..., num_tokens, token_dim)`` the way both code paths do.

    Python uses ``view(*shape[:-2], -1)`` and the ONNX encoder export uses
    ``flatten(start_dim=-2)``. Both are row-major over the trailing two axes,
    which is exactly ``numpy`` ``reshape``.
    """
    array = np.asarray(tokens)
    if array.shape[-2:] != (contract.max_num_tokens, contract.token_dim):
        raise ValueError(
            f"expected trailing shape ({contract.max_num_tokens}, {contract.token_dim}); "
            f"got {array.shape}"
        )
    return array.reshape(*array.shape[:-2], contract.total_dim)


def unflatten_tokens(flat: np.ndarray, contract: TokenContract) -> np.ndarray:
    """Inverse of :func:`flatten_tokens`."""
    array = np.asarray(flat)
    if array.shape[-1] != contract.total_dim:
        raise ValueError(f"expected trailing dim {contract.total_dim}; got {array.shape}")
    return array.reshape(*array.shape[:-1], contract.max_num_tokens, contract.token_dim)


@dataclass(frozen=True)
class LatentParityReport:
    """Machine-readable result from :func:`check_latent_parity`."""

    verdict: str
    errors: tuple[str, ...]
    notes: tuple[str, ...]
    frame_count: int
    total_dim: int
    on_lattice_fraction: float
    max_lattice_deviation: float
    residual_rms: float
    min_value: float
    max_value: float
    out_of_encoder_range_count: int

    #: Recorded latent is byte-comparable with a C++ local-encoder ``token_state``.
    ENCODER_EQUIVALENT = "deployment_encoder_equivalent"
    #: Recorded latent is the decoder input *including* an RL residual.
    RESIDUAL_PERTURBED = "residual_perturbed_decoder_input"
    #: Structurally unusable: wrong width, wrong rank, or non-finite values.
    OUT_OF_CONTRACT = "out_of_contract"

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def deployment_encoder_equivalent(self) -> bool:
        return self.verdict == self.ENCODER_EQUIVALENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "frame_count": self.frame_count,
            "total_dim": self.total_dim,
            "on_lattice_fraction": self.on_lattice_fraction,
            "max_lattice_deviation": self.max_lattice_deviation,
            "residual_rms": self.residual_rms,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "out_of_encoder_range_count": self.out_of_encoder_range_count,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


def check_latent_parity(
    motion_token: Any,
    *,
    contract: TokenContract = SONIC_RELEASE_TOKEN_CONTRACT,
    lattice_atol: float = 1e-6,
    require_encoder_equivalent: bool = False,
) -> LatentParityReport:
    """Classify a recorded ``action.motion_token`` array against the deploy contract.

    Args:
        motion_token: ``(frames, total_dim)`` array of recorded decoder-input
            latents. A single ``(total_dim,)`` frame is also accepted.
        contract: FSQ token geometry to test against.
        lattice_atol: Absolute tolerance for calling a value "on-lattice".
            Post-quantization FSQ output is exact in float32, so this only
            absorbs float64/float32 round-tripping, not a real residual.
        require_encoder_equivalent: When true, a residual-carrying latent is an
            error rather than a note. Use this for datasets that must be mixed
            with real teleop episodes recorded from the deploy encoder.

    Returns:
        A :class:`LatentParityReport`. ``errors`` is non-empty only for shape,
        finiteness, and (optionally) representation-mismatch failures; the
        representation itself is always reported through ``verdict``.
    """
    errors: list[str] = []
    notes: list[str] = []

    array = np.asarray(motion_token)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        return LatentParityReport(
            verdict=LatentParityReport.OUT_OF_CONTRACT,
            errors=(f"motion token must be 1D or 2D; got shape {np.asarray(motion_token).shape}",),
            notes=(),
            frame_count=0,
            total_dim=contract.total_dim,
            on_lattice_fraction=0.0,
            max_lattice_deviation=float("inf"),
            residual_rms=float("inf"),
            min_value=float("nan"),
            max_value=float("nan"),
            out_of_encoder_range_count=0,
        )
    if array.shape[1] != contract.total_dim:
        errors.append(
            f"motion token width {array.shape[1]} does not match the deployment "
            f"token_state size {contract.total_dim}"
        )
    if array.shape[0] < 1:
        errors.append("motion token has no frames")

    try:
        values = array.astype(np.float64)
    except (TypeError, ValueError):
        errors.append(f"motion token must be numeric; got dtype {array.dtype}")
        values = np.zeros((0, contract.total_dim), dtype=np.float64)

    if values.size and not np.isfinite(values).all():
        errors.append("motion token contains NaN or infinity")
        values = values[np.isfinite(values).all(axis=1)]

    if errors or not values.size:
        return LatentParityReport(
            verdict=LatentParityReport.OUT_OF_CONTRACT,
            errors=tuple(errors) or ("motion token had no usable frames",),
            notes=tuple(notes),
            frame_count=int(array.shape[0]),
            total_dim=contract.total_dim,
            on_lattice_fraction=0.0,
            max_lattice_deviation=float("inf"),
            residual_rms=float("inf"),
            min_value=float("nan"),
            max_value=float("nan"),
            out_of_encoder_range_count=0,
        )

    flat_levels = contract.flat_levels()
    deviation = np.empty_like(values)
    below = np.zeros(values.shape, dtype=bool)
    above = np.zeros(values.shape, dtype=bool)
    for level in np.unique(flat_levels):
        columns = np.flatnonzero(flat_levels == level)
        codes = fsq_level_code_values(int(level))
        column_values = values[:, columns]
        # Distance to the nearest reachable code value.
        deviation[:, columns] = np.min(
            np.abs(column_values[..., None] - codes[None, None, :]), axis=-1
        )
        below[:, columns] = column_values < codes[0] - lattice_atol
        above[:, columns] = column_values > codes[-1] + lattice_atol

    on_lattice = deviation <= lattice_atol
    on_lattice_fraction = float(on_lattice.mean())
    max_lattice_deviation = float(deviation.max())
    residual_rms = float(np.sqrt(np.mean(np.square(deviation))))
    out_of_range = below | above
    out_of_encoder_range_count = int(out_of_range.sum())

    # Representation is decided by the lattice alone. An off-range value is
    # *evidence about* the residual magnitude, not a separate representation:
    # a post-quantization residual with scale 1.0 and an unbounded policy output
    # routinely pushes a token past the reachable code range.
    if on_lattice.all():
        verdict = LatentParityReport.ENCODER_EQUIVALENT
        notes.append(
            "every value lies on the FSQ codebook lattice; this latent is "
            "representation-identical to a C++ local-encoder token_state"
        )
    else:
        verdict = LatentParityReport.RESIDUAL_PERTURBED
        notes.append(
            "values are off the FSQ codebook lattice; this is the decoder input carrying "
            "the policy latent residual, not the encoder's own output"
        )
    if out_of_encoder_range_count:
        notes.append(
            f"{out_of_encoder_range_count} values fall outside the reachable FSQ code range, "
            "which the encoder alone can never produce"
        )

    notes.append(
        f"on-lattice fraction {on_lattice_fraction:.6f}, "
        f"max lattice deviation {max_lattice_deviation:.6g}, "
        f"residual RMS {residual_rms:.6g}"
    )
    notes.append(
        f"value range [{float(values.min()):.6g}, {float(values.max()):.6g}] over "
        f"{values.shape[0]} frames x {values.shape[1]} dims"
    )

    if require_encoder_equivalent and verdict != LatentParityReport.ENCODER_EQUIVALENT:
        errors.append(
            f"latent representation is '{verdict}' but encoder-equivalent tokens were required; "
            "do not mix this episode with teleop data recorded from the deploy encoder"
        )

    return LatentParityReport(
        verdict=verdict,
        errors=tuple(errors),
        notes=tuple(notes),
        frame_count=int(values.shape[0]),
        total_dim=contract.total_dim,
        on_lattice_fraction=on_lattice_fraction,
        max_lattice_deviation=max_lattice_deviation,
        residual_rms=residual_rms,
        min_value=float(values.min()),
        max_value=float(values.max()),
        out_of_encoder_range_count=out_of_encoder_range_count,
    )


_TOP_LEVEL_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
_NESTED_KEY = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$")


def read_deploy_encoder_dimension(path: str | Path) -> int:
    """Read ``encoder.dimension`` from a C++ deploy observation config.

    Deliberately dependency-free (no PyYAML) so this cross-repository binding
    can be asserted from any environment, matching
    :mod:`gear_sonic.dataset_generation.scene_asset_preflight`.
    """
    text = Path(path).read_text(encoding="utf-8")
    in_encoder = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        top = _TOP_LEVEL_KEY.match(line)
        if top is not None:
            in_encoder = top.group(1) == "encoder"
            continue
        if not in_encoder:
            continue
        nested = _NESTED_KEY.match(line)
        if nested is not None and nested.group(1) == "dimension":
            try:
                return int(nested.group(2))
            except ValueError as exc:
                raise ValueError(
                    f"encoder.dimension in {path} is not an integer: {nested.group(2)!r}"
                ) from exc
    raise ValueError(f"no encoder.dimension found in {path}")


def deploy_observation_enabled(path: str | Path, name: str) -> bool:
    """Return whether ``observations[name].enabled`` is true in a deploy config."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [raw.split("#", 1)[0].rstrip() for raw in text.splitlines()]
    in_observations = False
    matched = False
    for line in lines:
        if not line.strip():
            continue
        top = _TOP_LEVEL_KEY.match(line)
        if top is not None:
            in_observations = top.group(1) == "observations"
            matched = False
            continue
        if not in_observations:
            continue
        stripped = line.strip()
        if stripped.startswith("- name:"):
            matched = stripped.split(":", 1)[1].strip().strip('"').strip("'") == name
            continue
        if matched and stripped.startswith("enabled:"):
            return stripped.split(":", 1)[1].strip().lower() == "true"
    return False


def describe_contract(contract: TokenContract = SONIC_RELEASE_TOKEN_CONTRACT) -> dict[str, Any]:
    """Summarise the token contract for provenance records."""
    unique_levels: Sequence[int] = sorted(set(contract.fsq_level_list))
    return {
        "max_num_tokens": contract.max_num_tokens,
        "token_dim": contract.token_dim,
        "total_dim": contract.total_dim,
        "fsq_levels": list(unique_levels),
        "flatten_order": "row_major_over_(num_tokens, token_dim)",
        "representation": "post_quantization_fsq_codes",
        "python_source": "UniversalTokenModule._last_full_latent_flat -> env._full_latent",
        "cpp_source": "encoder ONNX output 'encoded_tokens' -> token_state observation",
    }


def latent_parity_provenance(
    report: LatentParityReport,
    *,
    contract: TokenContract = SONIC_RELEASE_TOKEN_CONTRACT,
) -> dict[str, Any]:
    """Build the provenance block an exporter should embed alongside a dataset."""
    payload = describe_contract(contract)
    payload["parity_report"] = report.to_dict()
    return payload


def summarize_payload_latent(
    payload: Mapping[str, Any],
    *,
    key: str = "action_motion_token",
    contract: TokenContract = SONIC_RELEASE_TOKEN_CONTRACT,
    lattice_atol: float = 1e-6,
    require_encoder_equivalent: bool = False,
) -> LatentParityReport:
    """Run :func:`check_latent_parity` on a raw recorder payload."""
    if key not in payload:
        return LatentParityReport(
            verdict=LatentParityReport.OUT_OF_CONTRACT,
            errors=(f"payload is missing '{key}'",),
            notes=(),
            frame_count=0,
            total_dim=contract.total_dim,
            on_lattice_fraction=0.0,
            max_lattice_deviation=float("inf"),
            residual_rms=float("inf"),
            min_value=float("nan"),
            max_value=float("nan"),
            out_of_encoder_range_count=0,
        )
    return check_latent_parity(
        payload[key],
        contract=contract,
        lattice_atol=lattice_atol,
        require_encoder_equivalent=require_encoder_equivalent,
    )
