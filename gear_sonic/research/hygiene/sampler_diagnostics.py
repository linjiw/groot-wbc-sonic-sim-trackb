"""Where does SONIC's adaptive sampler actually spend training exposure?

WHY THIS MODULE EXISTS
----------------------
"Add a uniform floor to the sampler" is not an available intervention: SONIC already
ships the normalised uniform mixture, and it ships it at exactly ``a = 0.10``.

    ``motion_lib_base.py:3235-3238``  per-reset active-batch path::

        adp_sampling_active_prob = failure_based * (1 - uniform_sampling_rate)
                                   + uniform * uniform_sampling_rate

    ``motion_lib_base.py:3399-3402``  global clip-residency path (same blend).

    then ``:3239`` / ``:3403``  ``prob *= adp_samp_bin_weights``
    then ``:3240-3242`` / ``:3404``  renormalise.

``uniform_sampling_rate`` is read at ``motion_lib_base.py:2457`` and pinned at
``gear_sonic/config/manager_env/commands/terms/motion.yaml:26`` (``0.1``). The bin
weights are built at ``:2447-2451`` as
``w_i = (bin_length_i / mean_bin_length) / num_peer_bins_motion``, so ``w`` scales as
one over the number of bins in the clip and the per-MOTION weight sum is *nearly* a
constant -- the shipped comment at ``:2450`` calls this "each sequence is sampled equally".
It is only exactly equal for clips whose frame count is a multiple of ``bin_size``; the
remainder bin costs a ``k*bin_size + 1`` clip a third of its exposure, and on a realistic
length mix the t=0 prior already spans 1.49x across clips. See
:meth:`BinLayout.motion_weight_sums`.

The useful question is therefore not "is there a floor" but "what does the floor buy".
A uniform floor bounds each bin's probability FROM BELOW. It bounds nothing from above.
So the nominal 10% uniform mass can survive the bin-weight multiply essentially intact
while a single impossible clip still absorbs tens of times its fair share of exposure.
This module measures both halves of that sentence, and the ledger that turns the second
half into training cost (:func:`wasted_exposure`).

DESIGN RULE: NEVER REIMPLEMENT THE PROBABILITY MATH
---------------------------------------------------
:func:`sampling_distribution` is a thin wrapper that drives the REAL
``MotionLibBase.sync_and_compute_adaptive_sampling`` through an
``object.__new__(MotionLibBase)`` stub. The pattern comes from
``tests/research/test_adaptive_sampling_prob.py:18-32``, but that fixture stubs
``adp_samp_bin_weights`` to ones -- which is precisely the regime that never occurs in
training and that these diagnostics exist to probe. :func:`build_bins` supplies the real
length-derived weights instead. Because the probabilities come out of the shipped code
path, the diagnostics cannot drift away from the sampler they describe.

ENTROPY: TWO DIFFERENT NUMBERS, NEVER INTERCHANGEABLE
------------------------------------------------------
SONIC logs ``adp_samp/effective_num_bins = 1 / sum(p^2)`` at
``gear_sonic/envs/wrapper/manager_env_wrapper.py:1023``. That is the exponential of the
Renyi-2 (collision) entropy -- the inverse participation ratio -- NOT Shannon entropy.
:func:`effective_num_bins` reproduces it; :func:`normalized_shannon_entropy` computes the
Shannon quantity. Renyi-2 is strictly more sensitive to the heavy end of the
distribution, so on any skewed distribution ``effective_num_bins < exp(H_shannon)``, and
the gap grows with concentration. The two must never be quoted side by side as if they
were the same statistic; say which one a number is.

CPU only. No Isaac Lab, no GPU, no training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SAMPLER_DIAGNOSTICS_SCHEMA_VERSION = 1

# Release-configuration constants, all read back from the shipped yamls.
RELEASE_BIN_SIZE = 50  # motion.yaml:23
RELEASE_INIT_NUM_FAILURES = 1.0  # motion.yaml:25
RELEASE_UNIFORM_SAMPLING_RATE = 0.1  # motion.yaml:26
RELEASE_FAILURE_RATE_CAP = 200.0  # sonic_release.yaml:71 (motion.yaml default is 50.0)
DEFAULT_FAILURE_RATE_CAP = 50.0  # motion.yaml:30, when no exp overlay raises it

# Verbatim from scripts/research/dump_sampler_checkpoint_state.py -- realized sampling
# mass read out of a checkpoint is NOT executed-start-bin mass.
SAMPLING_MASS_CAVEAT = (
    "sampled_count/sampled_fraction are selected-target-bin mass before the "
    "pre_failure_sample_window shift, not executed-start-bin mass."
)

_DEGENERATE_DECOMPOSITION_TOL = 1e-24


def _as_float_array(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _probabilities(values: ArrayLike, name: str = "p") -> FloatArray:
    array = _as_float_array(values, name)
    if (array < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError(f"{name} must carry positive mass")
    return array / total


# ---------------------------------------------------------------------------
# 1. Bin construction -- motion_lib_base.py:2400-2451
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinLayout:
    """The sampler's bin decomposition of a motion bank.

    Reproduces ``MotionLibBase.init_adaptive_sampling`` at
    ``gear_sonic/utils/motion_lib/motion_lib_base.py:2400-2451``:

    * ``:2401-2407``  ``bin_starts = arange(0, num_frames, bin_size)``,
      ``bin_ends = min(bin_starts + bin_size, num_frames)`` -- so the final bin of a clip
      is a short remainder bin whenever ``num_frames % bin_size != 0``.
    * ``:2422-2423``  ``num_peer_bins`` is the clip's own bin count, broadcast to each of
      its bins.
    * ``:2447-2449``  ``bin_weights = bin_length / mean(bin_length)``.
    * ``:2450-2451``  when ``sequence_length_agnostic`` (the shipped default, motion.yaml:24)
      ``bin_weights /= num_peer_bins``.

    Weights are computed in torch with the shipped dtypes (long lengths divided by a
    float32 mean, then by a long peer count) and widened to float64 for reporting, so the
    values here are bit-for-bit the float32 values training uses.

    ``motion_lengths_frames`` are frame counts at the sampler's target fps (SONIC resamples
    30 -> 50 Hz at load); ``MotionLibBase`` derives them at ``:2355-2364``.
    """

    motion_lengths_frames: IntArray
    bin_starts: IntArray
    bin_ends: IntArray
    motion_index: IntArray
    num_peer_bins: IntArray
    bin_weights: FloatArray
    bin_size: int
    sequence_length_agnostic: bool

    @property
    def num_bins(self) -> int:
        return int(self.motion_index.size)

    @property
    def num_motions(self) -> int:
        return int(self.motion_lengths_frames.size)

    def motion_weight_sums(self) -> FloatArray:
        """Total bin weight per motion -- what "each sequence is sampled equally" delivers.

        Constant across motions ONLY when ``sequence_length_agnostic`` is on AND every clip
        is an exact multiple of ``bin_size``. The final remainder bin breaks it: a clip of
        ``k * bin_size + r`` frames has ``k + 1`` peer bins but only ``k * bin_size + r``
        frames of weight, so its sum is ``(k * bin_size + r) / (k + 1)`` instead of
        ``bin_size``. A 101-frame clip therefore carries 0.673x the weight of a 100-frame
        one -- a third less exposure for one extra frame of mocap.

        On a realistic 800-clip retargeted-mocap length mix only ~2% of clips are exact
        multiples of 50, and the t=0 prior already spans 1.49x between the least- and
        most-exposed clip (0.74x to 1.10x fair share), biased against SHORT clips. Quote
        that as the floor of any fairness claim about this sampler; it is present before
        any adaptive signal exists. Pinned by
        ``tests/research/test_hygiene_sampler_diagnostics.py::
        test_release_prior_is_not_exactly_fair_across_motions``.
        """
        sums = np.zeros(self.num_motions, dtype=np.float64)
        np.add.at(sums, self.motion_index, self.bin_weights)
        return sums

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_bins": self.num_bins,
            "num_motions": self.num_motions,
            "bin_size": self.bin_size,
            "sequence_length_agnostic": self.sequence_length_agnostic,
            "bin_weight_min": float(self.bin_weights.min()),
            "bin_weight_max": float(self.bin_weights.max()),
            "bin_weight_max_over_min": float(self.bin_weights.max() / self.bin_weights.min()),
            "motion_weight_sum_min": float(self.motion_weight_sums().min()),
            "motion_weight_sum_max": float(self.motion_weight_sums().max()),
        }


def build_bins(
    motion_lengths_frames: ArrayLike,
    bin_size: int = RELEASE_BIN_SIZE,
    sequence_length_agnostic: bool = True,
) -> BinLayout:
    """Build the sampler's bins and bin weights for a bank of clip lengths.

    Faithful to ``motion_lib_base.py:2400-2451``; see :class:`BinLayout` for the
    line-by-line correspondence. Lengths are frame counts at the sampler's target fps.
    """
    lengths = np.asarray(motion_lengths_frames)
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError(
            f"motion_lengths_frames must be a non-empty 1-D array, got {lengths.shape}"
        )
    if not np.issubdtype(lengths.dtype, np.integer):
        if not np.all(np.isfinite(lengths)) or not np.all(lengths == np.floor(lengths)):
            raise ValueError("motion_lengths_frames must be whole frame counts")
        lengths = lengths.astype(np.int64)
    lengths = lengths.astype(np.int64)
    if (lengths <= 0).any():
        raise ValueError("motion_lengths_frames must all be positive")
    if int(bin_size) <= 0:
        raise ValueError(f"bin_size must be positive, got {bin_size}")
    bin_size = int(bin_size)

    starts: list[torch.Tensor] = []
    ends: list[torch.Tensor] = []
    motion_ids: list[torch.Tensor] = []
    peers: list[torch.Tensor] = []
    for motion_id, num_frames in enumerate(lengths.tolist()):
        frames = torch.tensor(num_frames, dtype=torch.long)
        bin_starts = torch.arange(0, frames, bin_size, dtype=torch.long)
        bin_ends = torch.minimum(bin_starts + bin_size, frames)
        num_bins = len(bin_starts)
        starts.append(bin_starts)
        ends.append(bin_ends)
        motion_ids.append(torch.full((num_bins,), motion_id, dtype=torch.long))
        peers.append(torch.full((num_bins,), num_bins, dtype=torch.long))

    bin_starts_all = torch.cat(starts)
    bin_ends_all = torch.cat(ends)
    motion_index = torch.cat(motion_ids)
    num_peer_bins = torch.cat(peers)

    # :2447-2451 -- long lengths / float32 mean, then / long peer counts.
    bin_motion_length = bin_ends_all - bin_starts_all
    bin_weights = bin_motion_length / bin_motion_length.float().mean()
    if sequence_length_agnostic:
        bin_weights = bin_weights / num_peer_bins

    return BinLayout(
        motion_lengths_frames=lengths,
        bin_starts=bin_starts_all.numpy().astype(np.int64),
        bin_ends=bin_ends_all.numpy().astype(np.int64),
        motion_index=motion_index.numpy().astype(np.int64),
        num_peer_bins=num_peer_bins.numpy().astype(np.int64),
        bin_weights=bin_weights.double().numpy(),
        bin_size=bin_size,
        sequence_length_agnostic=bool(sequence_length_agnostic),
    )


# ---------------------------------------------------------------------------
# 2. The sampler wrapper -- drives the real code path, never a copy of it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerConfig:
    """The four knobs the shipped failure-rate sampler reads.

    ``uniform_rate``          ``adaptive_sampling.uniform_sampling_rate`` (motion.yaml:26 = 0.1).
    ``failure_rate_cap``      ``adp_samp_failure_rate_max_over_mean`` (motion.yaml:30 = 50.0;
                              the release/bones-seed overlays set 200). It clips the failure
                              rate at ``mean * cap`` before normalisation (``:3216-3219``).
                              Failure rates live in [0, 1], so this bound only binds when the
                              mean failure rate is below ``1 / cap`` -- an almost-solved bank.
    ``max_prob_per_bin``      ``motion_lib_base.py:2461``, read at ``:3298-3316``.
    ``max_prob_per_motion``   ``motion_lib_base.py:2462``, read at ``:3318-3346``.

    Both concentration caps are ``None`` in EVERY shipped yaml, and ``None`` makes the whole
    constraint block early-return at ``:3285-3292``. ``"auto"`` resolves to
    ``failure_rate_cap / num_active_{bins,motions}``, which at cap 200 is far above any
    realistic share and is therefore also inert. Only an explicit small multiple of fair
    share binds.
    """

    uniform_rate: float = RELEASE_UNIFORM_SAMPLING_RATE
    failure_rate_cap: float = RELEASE_FAILURE_RATE_CAP
    max_prob_per_bin: float | str | None = None
    max_prob_per_motion: float | str | None = None

    def validate(self) -> None:
        if not 0.0 <= float(self.uniform_rate) <= 1.0:
            raise ValueError(f"uniform_rate must be in [0, 1], got {self.uniform_rate}")
        if not math.isfinite(float(self.failure_rate_cap)) or float(self.failure_rate_cap) <= 0:
            raise ValueError(
                f"failure_rate_cap must be positive and finite, got {self.failure_rate_cap}"
            )
        for name in ("max_prob_per_bin", "max_prob_per_motion"):
            value = getattr(self, name)
            if value is None or value == "auto":
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be None, 'auto', or a finite number, got {value!r}")

    def replace(self, **kwargs: Any) -> "SamplerConfig":
        fields = {
            "uniform_rate": self.uniform_rate,
            "failure_rate_cap": self.failure_rate_cap,
            "max_prob_per_bin": self.max_prob_per_bin,
            "max_prob_per_motion": self.max_prob_per_motion,
        }
        unknown = set(kwargs) - set(fields)
        if unknown:
            raise ValueError(f"unknown SamplerConfig fields: {sorted(unknown)}")
        fields.update(kwargs)
        return SamplerConfig(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uniform_rate": float(self.uniform_rate),
            "failure_rate_cap": float(self.failure_rate_cap),
            "max_prob_per_bin": self.max_prob_per_bin,
            "max_prob_per_motion": self.max_prob_per_motion,
        }


def release_prior_counts(
    layout: BinLayout, init_num_failures: float = RELEASE_INIT_NUM_FAILURES
) -> tuple[FloatArray, FloatArray]:
    """The sampler's t=0 state: ``num_failures == num_episodes == init_num_failures``.

    ``motion_lib_base.py:2463-2471`` initialises both counters to ``init_num_failures``
    (motion.yaml:25 = 1), so every bin starts at failure rate 1.0 and the failure-based
    component is exactly uniform over bins before any episode is played.
    """
    if init_num_failures <= 0:
        raise ValueError(
            "init_num_failures must be positive here; the 0/0 regime is exercised by "
            "tests/research/test_adaptive_sampling_prob.py, not by these diagnostics"
        )
    counts = np.full(layout.num_bins, float(init_num_failures), dtype=np.float64)
    return counts.copy(), counts.copy()


def counts_from_failure_rates(
    failure_rates: ArrayLike, num_episodes: float | ArrayLike = 1.0
) -> tuple[FloatArray, FloatArray]:
    """Turn per-bin failure rates into the ``(num_failures, num_episodes)`` the sampler stores.

    The sampler only ever sees the ratio (``:2952-2956``), so ``failures = rate * episodes``
    reproduces any target rate exactly.
    """
    rates = _as_float_array(failure_rates, "failure_rates")
    if (rates < 0).any() or (rates > 1).any():
        raise ValueError("failure_rates must lie in [0, 1]")
    episodes = np.broadcast_to(np.asarray(num_episodes, dtype=np.float64), rates.shape).copy()
    if (episodes <= 0).any():
        raise ValueError("num_episodes must be positive")
    return rates * episodes, episodes


def _sampler_stub(
    layout: BinLayout,
    num_failures: FloatArray,
    num_episodes: FloatArray,
    config: SamplerConfig,
    active_bins: IntArray,
) -> MotionLibBase:
    """Minimal ``MotionLibBase`` exposing exactly what the probability path reads.

    Follows ``tests/research/test_adaptive_sampling_prob.py:18-32``, except that
    ``adp_samp_bin_weights`` carries the REAL length-derived weights and ``adp_samp_bins``
    carries the real motion ids, so the per-motion cap path at ``:3334`` has something
    truthful to aggregate over. ``adaptive_sampling_cfg = {}`` selects the release
    behaviour: ``signal == "failure_rate"``, no evidence decay, no failure-rate decay.
    """
    stub = object.__new__(MotionLibBase)
    stub.use_adaptive_sampling = True
    stub.adaptive_sampling_cfg = {}
    stub.adp_samp_num_episodes = torch.as_tensor(num_episodes, dtype=torch.float32)
    stub.adp_samp_num_failures = torch.as_tensor(num_failures, dtype=torch.float32)
    stub.adp_samp_active_motion_bins = torch.as_tensor(active_bins, dtype=torch.long)
    stub.adp_samp_failure_rate_max_over_mean = float(config.failure_rate_cap)
    stub.uniform_sampling_rate = float(config.uniform_rate)
    stub.adp_samp_bin_weights = torch.as_tensor(layout.bin_weights, dtype=torch.float32)
    stub.max_prob_per_bin_cfg = config.max_prob_per_bin
    stub.max_prob_per_motion_cfg = config.max_prob_per_motion
    stub.adp_samp_bins = torch.stack(
        [
            torch.as_tensor(layout.motion_index, dtype=torch.long),
            torch.as_tensor(layout.bin_starts, dtype=torch.long),
            torch.as_tensor(layout.bin_ends, dtype=torch.long),
        ],
        dim=1,
    )
    # Only read when use_failure_rate_decay is on (off in release), but keep it truthful.
    new_motion_mask = np.zeros(layout.num_bins, dtype=bool)
    new_motion_mask[np.flatnonzero(np.diff(layout.motion_index, prepend=-1) != 0)] = True
    stub.adp_samp_bin_new_motion_mask = torch.as_tensor(new_motion_mask)
    return stub


def _resolve_counts(
    layout: BinLayout,
    failure_rates: ArrayLike | None,
    num_failures: ArrayLike | None,
    num_episodes: ArrayLike | None,
) -> tuple[FloatArray, FloatArray]:
    if failure_rates is not None:
        if num_failures is not None:
            raise ValueError("pass either failure_rates or num_failures, not both")
        failures, episodes = counts_from_failure_rates(
            failure_rates, 1.0 if num_episodes is None else num_episodes
        )
    elif num_failures is not None:
        failures = _as_float_array(num_failures, "num_failures")
        if num_episodes is None:
            raise ValueError("num_episodes is required when num_failures is given")
        episodes = _as_float_array(num_episodes, "num_episodes")
    else:
        failures, episodes = release_prior_counts(layout)
    if failures.shape != (layout.num_bins,) or episodes.shape != (layout.num_bins,):
        raise ValueError(
            f"counts must have one entry per bin ({layout.num_bins}), got "
            f"{failures.shape} / {episodes.shape}"
        )
    return failures, episodes


def _resolve_active_bins(layout: BinLayout, active_bins: ArrayLike | None) -> IntArray:
    if active_bins is None:
        return np.arange(layout.num_bins, dtype=np.int64)
    active = np.asarray(active_bins, dtype=np.int64)
    if active.ndim != 1 or active.size == 0:
        raise ValueError("active_bins must be a non-empty 1-D index array")
    if (active < 0).any() or (active >= layout.num_bins).any():
        raise ValueError("active_bins contains out-of-range bin indices")
    if np.unique(active).size != active.size:
        raise ValueError("active_bins must not repeat a bin")
    return active


def _run_sampler(
    layout: BinLayout,
    *,
    failure_rates: ArrayLike | None = None,
    num_failures: ArrayLike | None = None,
    num_episodes: ArrayLike | None = None,
    config: SamplerConfig = SamplerConfig(),
    active_bins: ArrayLike | None = None,
) -> MotionLibBase:
    """Drive the real ``sync_and_compute_adaptive_sampling`` and hand back the stub."""
    config.validate()
    failures, episodes = _resolve_counts(layout, failure_rates, num_failures, num_episodes)
    active = _resolve_active_bins(layout, active_bins)
    stub = _sampler_stub(layout, failures, episodes, config, active)
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    return stub


def sampling_distribution(
    layout: BinLayout,
    *,
    failure_rates: ArrayLike | None = None,
    num_failures: ArrayLike | None = None,
    num_episodes: ArrayLike | None = None,
    config: SamplerConfig = SamplerConfig(),
    active_bins: ArrayLike | None = None,
) -> FloatArray:
    """Per-bin sampling probabilities, computed by SONIC's own sampler.

    Calls ``MotionLibBase.sync_and_compute_adaptive_sampling`` on a stub holding the real
    bin weights, so the blend (``:3235-3238``), the weight multiply (``:3239``), the
    renormalisation (``:3240-3242``) and both optional concentration caps (``:3285-3346``)
    are the shipped implementations, not a re-derivation. Nothing here reimplements them.

    With no counts supplied the release prior (``init_num_failures = 1``) is used.
    Returns a float64 array over ``active_bins`` (all bins by default), summing to 1.
    """
    stub = _run_sampler(
        layout,
        failure_rates=failure_rates,
        num_failures=num_failures,
        num_episodes=num_episodes,
        config=config,
        active_bins=active_bins,
    )
    return stub.adp_sampling_active_prob.detach().double().numpy()


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------


def normalized_shannon_entropy(p: ArrayLike) -> float:
    """``-sum(p log p) / log(N)`` -- 1.0 for the uniform distribution, 0.0 for a point mass.

    This is SHANNON entropy. It is not the quantity SONIC logs as
    ``adp_samp/effective_num_bins`` (see :func:`effective_num_bins`); the two answer
    different questions and must never be presented as the same statistic.
    """
    prob = _probabilities(p)
    if prob.size == 1:
        return 0.0
    entropy = float(-(prob * np.log(np.clip(prob, 1e-300, None))).sum())
    return entropy / math.log(prob.size)


def effective_num_bins(p: ArrayLike) -> float:
    """``1 / sum(p^2)`` -- the number SONIC logs at ``manager_env_wrapper.py:1023``.

    This is the inverse participation ratio, i.e. ``exp(H_2)`` for the Renyi-2 (collision)
    entropy -- NOT ``exp(H_shannon)``. It weights the heavy end of the distribution more
    aggressively, so on any non-uniform ``p`` it is strictly smaller than
    ``exp(H_shannon)``. Quoting one as if it were the other understates or overstates
    concentration depending on which way the substitution went; do not mix them.
    """
    prob = _probabilities(p)
    return float(1.0 / np.square(prob).sum())


@dataclass(frozen=True)
class UniformMassDecomposition:
    """How much of the nominal uniform rate ``a`` survives the bin-weight multiply.

    The post-renormalisation distribution is
    ``p_i = [(1-a) f_i + a/N] * w_i / Z``. The numerator is linear in the mixture, so
    ``p(a) = x * p(0) + y * p(1)`` with ``x + y = 1``, and ``y`` is exactly the share of
    final mass contributed by the uniform component. All three distributions come out of
    the real sampler; ``y`` is recovered by a two-component least-squares fit, so no part
    of the probability math is re-derived here.

    ``residual_l1`` is the L1 error of that fit; it is ~0 whenever the identity holds.

    ``caps_active`` is measured, not inferred: it says whether ``max_prob_per_bin`` /
    ``max_prob_per_motion`` actually changed the distribution. When they did, the identity
    above no longer holds -- the caps are a nonlinear projection applied AFTER the blend --
    and ``realized_mass`` degrades to a least-squares projection onto the two-component
    span. Do not read it as a uniform share in that case. A small ``residual_l1`` is not
    reassurance here: a per-motion cap is piecewise linear and can pin the same motion at
    the same ceiling for every ``a``, which fits well and means nothing.

    ``degenerate`` marks the case ``p(0) == p(1)`` (every bin at the same failure rate, e.g.
    the release prior). There the fit is rank-deficient but the answer is analytically
    exactly ``a``, which is what is returned.
    """

    nominal_rate: float
    realized_mass: float
    residual_l1: float
    degenerate: bool
    caps_active: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal_rate": self.nominal_rate,
            "realized_mass": self.realized_mass,
            "realized_over_nominal": (
                self.realized_mass / self.nominal_rate if self.nominal_rate > 0 else math.nan
            ),
            "residual_l1": self.residual_l1,
            "degenerate": self.degenerate,
            "caps_active": self.caps_active,
        }


def uniform_mass_decomposition(
    layout: BinLayout,
    *,
    failure_rates: ArrayLike | None = None,
    num_failures: ArrayLike | None = None,
    num_episodes: ArrayLike | None = None,
    config: SamplerConfig = SamplerConfig(),
    active_bins: ArrayLike | None = None,
) -> UniformMassDecomposition:
    """Decompose the realized distribution into its failure-based and uniform components."""
    kwargs = {
        "failure_rates": failure_rates,
        "num_failures": num_failures,
        "num_episodes": num_episodes,
        "active_bins": active_bins,
    }
    p_mix = sampling_distribution(layout, config=config, **kwargs)
    p_failure = sampling_distribution(layout, config=config.replace(uniform_rate=0.0), **kwargs)
    p_uniform = sampling_distribution(layout, config=config.replace(uniform_rate=1.0), **kwargs)
    caps_active = False
    if config.max_prob_per_bin is not None or config.max_prob_per_motion is not None:
        uncapped = config.replace(max_prob_per_bin=None, max_prob_per_motion=None)
        p_uncapped = sampling_distribution(layout, config=uncapped, **kwargs)
        caps_active = bool(np.abs(p_mix - p_uncapped).max() > 1e-9)

    nominal = float(config.uniform_rate)
    basis = p_uniform - p_failure
    denominator = float(basis @ basis)
    if denominator <= _DEGENERATE_DECOMPOSITION_TOL:
        # p(0) == p(1): the failure-based component is already the uniform one, so the
        # blend is a no-op and the uniform share is exactly the nominal rate.
        return UniformMassDecomposition(nominal, nominal, 0.0, True, caps_active)
    residual_vector = p_mix - p_failure
    realized = float((residual_vector @ basis) / denominator)
    residual = float(np.abs(residual_vector - realized * basis).sum())
    return UniformMassDecomposition(nominal, realized, residual, False, caps_active)


def realized_uniform_mass(
    layout: BinLayout,
    *,
    failure_rates: ArrayLike | None = None,
    num_failures: ArrayLike | None = None,
    num_episodes: ArrayLike | None = None,
    config: SamplerConfig = SamplerConfig(),
    active_bins: ArrayLike | None = None,
) -> float:
    """Share of post-renormalisation mass attributable to the uniform component.

    Compare against ``config.uniform_rate``: the gap is how much of the nominal floor the
    bin-weight multiply gave or took. See :class:`UniformMassDecomposition` for the exact
    identity and its failure mode under concentration caps.
    """
    return uniform_mass_decomposition(
        layout,
        failure_rates=failure_rates,
        num_failures=num_failures,
        num_episodes=num_episodes,
        config=config,
        active_bins=active_bins,
    ).realized_mass


@dataclass(frozen=True)
class FailureRateCapDiagnostic:
    """Does ``adp_samp_failure_rate_max_over_mean`` actually clip anything?

    ``motion_lib_base.py:3216-3219`` clips the per-bin failure rate at
    ``mean(failure_rate) * cap``. Failure rates are bounded by 1, so the clip can only bind
    when ``mean * cap < 1``, i.e. when the mean failure rate is below ``1 / cap``. At the
    release cap of 200 that means a mean failure rate below 0.005 -- an almost-solved bank.
    Anywhere else this "heavy-tail guard" is inert, which is why it does not stop a single
    impossible clip from dominating.

    ``binds`` is measured, not assumed: it compares the shipped distribution against the
    same distribution computed with an effectively infinite cap. It is deliberately NOT
    ``num_bins_clipped > 0``: clipping every positive-signal bin to the same constant is
    undone by the renormalisation that follows, so a clip can fire and still change nothing.
    """

    cap: float
    mean_failure_rate: float
    upper_bound: float
    mean_failure_rate_for_binding: float
    num_bins_clipped: int
    binds: bool
    max_abs_prob_delta_vs_uncapped: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cap": self.cap,
            "mean_failure_rate": self.mean_failure_rate,
            "upper_bound": self.upper_bound,
            "mean_failure_rate_for_binding": self.mean_failure_rate_for_binding,
            "num_bins_clipped": self.num_bins_clipped,
            "binds": self.binds,
            "max_abs_prob_delta_vs_uncapped": self.max_abs_prob_delta_vs_uncapped,
        }


def failure_rate_cap_diagnostic(
    layout: BinLayout,
    *,
    failure_rates: ArrayLike | None = None,
    num_failures: ArrayLike | None = None,
    num_episodes: ArrayLike | None = None,
    config: SamplerConfig = SamplerConfig(),
    active_bins: ArrayLike | None = None,
    uncapped_value: float = 1e12,
) -> FailureRateCapDiagnostic:
    """Measure whether the mean x cap failure-rate clip changes the distribution at all."""
    kwargs = {
        "failure_rates": failure_rates,
        "num_failures": num_failures,
        "num_episodes": num_episodes,
        "active_bins": active_bins,
    }
    stub = _run_sampler(layout, config=config, **kwargs)
    # Rates are the sampler's own (:2952-2958, then :3195-3197); only the bound is mirrored.
    active_rate = stub.adp_samp_active_failure_rate.detach().double().numpy()
    mean_rate = float(active_rate.mean())
    cap = float(config.failure_rate_cap)
    upper_bound = mean_rate * cap
    p_capped = stub.adp_sampling_active_prob.detach().double().numpy()
    p_uncapped = sampling_distribution(
        layout, config=config.replace(failure_rate_cap=uncapped_value), **kwargs
    )
    delta = float(np.abs(p_capped - p_uncapped).max())
    return FailureRateCapDiagnostic(
        cap=cap,
        mean_failure_rate=mean_rate,
        upper_bound=upper_bound,
        mean_failure_rate_for_binding=1.0 / cap,
        num_bins_clipped=int((active_rate > upper_bound).sum()),
        binds=bool(delta > 1e-9),
        max_abs_prob_delta_vs_uncapped=delta,
    )


@dataclass(frozen=True)
class MotionExposure:
    """Per-motion share of sampling mass, and how far it is from fair share.

    ``fair_share = 1 / num_motions``. ``ratio = share / fair_share`` is the number the
    uniform floor does NOT bound: the floor guarantees ``share >= a / num_motions``-ish from
    below and says nothing about how large ``ratio`` may grow.
    """

    share: FloatArray
    ratio: FloatArray
    fair_share: float
    top1_index: int
    top1_share: float
    top1_ratio: float
    motion_keys: tuple[str, ...] | None

    @property
    def num_motions(self) -> int:
        return int(self.share.size)

    @property
    def top1_key(self) -> str | None:
        return None if self.motion_keys is None else self.motion_keys[self.top1_index]

    def share_for(self, key: str) -> float:
        """Sampling mass on one motion key."""
        if self.motion_keys is None:
            raise ValueError("this MotionExposure was built without motion_keys")
        return float(self.share[self.motion_keys.index(key)])

    def top_k(self, k: int = 10) -> list[dict[str, Any]]:
        order = np.argsort(self.share)[::-1][: max(int(k), 0)]
        return [
            {
                "motion_index": int(index),
                "motion_key": None if self.motion_keys is None else self.motion_keys[index],
                "share": float(self.share[index]),
                "share_over_fair": float(self.ratio[index]),
            }
            for index in order
        ]

    def to_dict(self, top_k_count: int = 10) -> dict[str, Any]:
        """Scalar summary plus the heaviest clips.

        The full per-motion arrays stay on the dataclass; a 4950-clip bank does not belong
        inline in a report row.
        """
        return {
            "num_motions": self.num_motions,
            "fair_share": self.fair_share,
            "top1_motion_index": self.top1_index,
            "top1_motion_key": self.top1_key,
            "top1_share": self.top1_share,
            "top1_share_over_fair": self.top1_ratio,
            "share_over_fair_p99": float(np.percentile(self.ratio, 99.0)),
            "num_motions_over_10x_fair": int((self.ratio > 10.0).sum()),
            "top_k": self.top_k(top_k_count),
        }


def per_motion_exposure(
    p: ArrayLike,
    motion_index: ArrayLike,
    motion_keys: Sequence[str] | None = None,
    num_motions: int | None = None,
) -> MotionExposure:
    """Aggregate per-bin sampling mass to per-motion exposure.

    ``motion_index`` is the bin -> motion map (``BinLayout.motion_index``, or a checkpoint's
    ``adp_samp_bin_motion_ids``). Motions with no bins in ``p`` get share 0.
    """
    prob = _probabilities(p)
    index = np.asarray(motion_index, dtype=np.int64)
    if index.shape != prob.shape:
        raise ValueError(f"motion_index shape {index.shape} does not match p shape {prob.shape}")
    if (index < 0).any():
        raise ValueError("motion_index must be nonnegative")
    total_motions = int(index.max()) + 1 if num_motions is None else int(num_motions)
    if total_motions <= 0 or total_motions <= int(index.max()):
        raise ValueError(f"num_motions={num_motions} is too small for motion_index")
    if motion_keys is not None and len(motion_keys) != total_motions:
        raise ValueError(
            f"motion_keys has {len(motion_keys)} entries but there are {total_motions} motions"
        )

    share = np.zeros(total_motions, dtype=np.float64)
    np.add.at(share, index, prob)
    fair = 1.0 / total_motions
    ratio = share / fair
    top1 = int(np.argmax(share))
    return MotionExposure(
        share=share,
        ratio=ratio,
        fair_share=fair,
        top1_index=top1,
        top1_share=float(share[top1]),
        top1_ratio=float(ratio[top1]),
        motion_keys=None if motion_keys is None else tuple(motion_keys),
    )


@dataclass(frozen=True)
class WastedExposure:
    """The ledger that turns a screen verdict into training cost.

    ``wasted_fraction`` is the share of all sampling mass spent on clips the dynamic
    feasibility screen flagged. It is directly comparable to ``flagged_bank_fraction`` (the
    share of clips flagged): a sampler that ignored feasibility would spend the two equally,
    so ``concentration_ratio > 1`` is exposure actively pulled toward clips no controller can
    track.
    """

    wasted_fraction: float
    flagged_motion_count: int
    flagged_bank_fraction: float
    concentration_ratio: float
    unknown_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wasted_fraction": self.wasted_fraction,
            "flagged_motion_count": self.flagged_motion_count,
            "flagged_bank_fraction": self.flagged_bank_fraction,
            "concentration_ratio": self.concentration_ratio,
            "unknown_key_count": len(self.unknown_keys),
            "unknown_keys": list(self.unknown_keys[:20]),
        }


def wasted_exposure(
    exposure: MotionExposure,
    infeasible_keys: Iterable[str],
    strict: bool = True,
) -> WastedExposure:
    """Fraction of sampling mass spent on screen-flagged clips.

    ``infeasible_keys`` are motion keys (clip filename stems). ``strict`` raises when a key
    is not in the bank -- a silent typo would understate waste, which is the one direction
    this ledger must not fail in. Set ``strict=False`` when the screen legitimately covers a
    superset of the training bank; the unmatched keys are then reported, not dropped.
    """
    if exposure.motion_keys is None:
        raise ValueError("wasted_exposure needs a MotionExposure built with motion_keys")
    lookup = {key: position for position, key in enumerate(exposure.motion_keys)}
    requested = list(dict.fromkeys(str(key) for key in infeasible_keys))
    matched = [lookup[key] for key in requested if key in lookup]
    unknown = tuple(key for key in requested if key not in lookup)
    if unknown and strict:
        raise ValueError(
            f"{len(unknown)} flagged key(s) are not in the bank, e.g. {unknown[:5]}; "
            "pass strict=False to report them instead"
        )
    wasted = float(exposure.share[matched].sum()) if matched else 0.0
    flagged_bank_fraction = len(matched) / exposure.num_motions
    ratio = wasted / flagged_bank_fraction if flagged_bank_fraction > 0 else math.nan
    return WastedExposure(
        wasted_fraction=wasted,
        flagged_motion_count=len(matched),
        flagged_bank_fraction=flagged_bank_fraction,
        concentration_ratio=ratio,
        unknown_keys=unknown,
    )


# ---------------------------------------------------------------------------
# 4. Realized exposure from a training checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointSamplerState:
    """Realized sampler state read out of a training checkpoint.

    CAVEAT, carried verbatim from ``scripts/research/dump_sampler_checkpoint_state.py``::

        sampled_count/sampled_fraction are selected-target-bin mass before the
        pre_failure_sample_window shift, not executed-start-bin mass.

    So ``realized_bin_share`` / ``realized_motion_share`` answer "which bin did the sampler
    pick as the target", not "which frames did the policy actually start from". The
    ``pre_failure_sample_window`` shift (200 frames in the release config) moves the executed
    start earlier, usually into a preceding bin of the same clip -- which is why the
    MOTION-level aggregate is the safer of the two to quote. The remaining upstream caveats
    (failure-rate decay assumption, active-subset mean for the clip bound, missing weights in
    legacy checkpoints) are preserved in ``caveats``.
    """

    checkpoint_path: str | None
    checkpoint_sha256: str | None
    global_step: int | None
    adaptive_state_present: bool
    num_bins: int
    num_motions: int | None
    motion_index: IntArray | None
    motion_keys: tuple[str, ...] | None
    num_episodes: FloatArray
    num_failures: FloatArray
    failure_rate: FloatArray
    bin_weights: FloatArray | None
    bin_draw_counts: IntArray | None
    realized_bin_share: FloatArray | None
    realized_motion_share: FloatArray | None
    recomputed_bin_prob_unweighted: FloatArray
    sampling_mass_semantics: str | None
    caveats: tuple[str, ...]

    def realized_motion_exposure(self) -> MotionExposure:
        """Per-motion exposure from the checkpoint's realized draw counts."""
        if self.realized_bin_share is None or self.motion_index is None:
            raise ValueError(
                "checkpoint has no bin draw counts or no bin->motion map; realized exposure "
                "is only available for ZPD-schema checkpoints"
            )
        return per_motion_exposure(
            self.realized_bin_share,
            self.motion_index,
            motion_keys=self.motion_keys,
            num_motions=self.num_motions,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "global_step": self.global_step,
            "adaptive_state_present": self.adaptive_state_present,
            "num_bins": self.num_bins,
            "num_motions": self.num_motions,
            "failure_rate_mean": float(self.failure_rate.mean()) if self.num_bins else math.nan,
            "failure_rate_max": float(self.failure_rate.max()) if self.num_bins else math.nan,
            "has_realized_draw_counts": self.bin_draw_counts is not None,
            "sampling_mass_semantics": self.sampling_mass_semantics,
            "caveats": list(self.caveats),
        }
        if self.realized_bin_share is not None and self.motion_index is not None:
            payload["realized_motion_exposure"] = self.realized_motion_exposure().to_dict()
        return payload


def summarize_checkpoint_sampler_state(
    state: dict[str, Any], **kwargs: Any
) -> CheckpointSamplerState:
    """Structure one already-extracted checkpoint state dict.

    Delegates every derived quantity to
    ``scripts.research.dump_sampler_checkpoint_state.build_sampler_state_summary`` so there
    is exactly one implementation of the checkpoint semantics in the repo; this only
    reshapes its output into arrays and aggregates bins to motions via
    ``adp_samp_bin_motion_ids``. ``kwargs`` are forwarded (``init_num_failures``,
    ``failure_rate_cap``, ``uniform_sampling_rate``, ``prior_domination_threshold``).
    """
    from scripts.research.dump_sampler_checkpoint_state import build_sampler_state_summary

    summary = build_sampler_state_summary(state, **kwargs)
    caveats = tuple(summary.get("caveats", ()))
    if not summary.get("adaptive_state_present"):
        empty = np.zeros(0, dtype=np.float64)
        return CheckpointSamplerState(
            checkpoint_path=summary.get("checkpoint_path"),
            checkpoint_sha256=summary.get("checkpoint_sha256"),
            global_step=summary.get("global_step"),
            adaptive_state_present=False,
            num_bins=0,
            num_motions=None,
            motion_index=None,
            motion_keys=None,
            num_episodes=empty,
            num_failures=empty,
            failure_rate=empty,
            bin_weights=None,
            bin_draw_counts=None,
            realized_bin_share=None,
            realized_motion_share=None,
            recomputed_bin_prob_unweighted=empty,
            sampling_mass_semantics=summary.get("sampling_mass_semantics"),
            caveats=caveats,
        )

    bins = summary["bins"]
    num_episodes = np.array([row["num_episodes"] for row in bins], dtype=np.float64)
    num_failures = np.array([row["num_failures"] for row in bins], dtype=np.float64)
    failure_rate = np.array([row["failure_rate"] for row in bins], dtype=np.float64)
    recomputed = np.array([row["recomputed_prob_unweighted"] for row in bins], dtype=np.float64)
    weights = (
        np.array([row["bin_weight"] for row in bins], dtype=np.float64)
        if "bin_weight" in bins[0]
        else None
    )
    draw_counts = (
        np.array([row["sampled_count"] for row in bins], dtype=np.int64)
        if "sampled_count" in bins[0]
        else None
    )
    realized_bin_share = (
        np.array([row["sampled_fraction"] for row in bins], dtype=np.float64)
        if draw_counts is not None
        else None
    )

    motion_ids = state.get("adp_samp_bin_motion_ids")
    motion_index = np.asarray(motion_ids, dtype=np.int64) if motion_ids is not None else None
    motion_keys_raw = state.get("adp_samp_motion_data_keys")
    motion_keys = tuple(str(key) for key in motion_keys_raw) if motion_keys_raw else None
    num_motions = len(motion_keys) if motion_keys is not None else None
    if num_motions is None and motion_index is not None:
        num_motions = int(motion_index.max()) + 1

    realized_motion_share = None
    if realized_bin_share is not None and motion_index is not None and num_motions:
        realized_motion_share = np.zeros(num_motions, dtype=np.float64)
        np.add.at(realized_motion_share, motion_index, realized_bin_share)

    return CheckpointSamplerState(
        checkpoint_path=summary.get("checkpoint_path"),
        checkpoint_sha256=summary.get("checkpoint_sha256"),
        global_step=summary.get("global_step"),
        adaptive_state_present=True,
        num_bins=int(summary["num_bins"]),
        num_motions=num_motions,
        motion_index=motion_index,
        motion_keys=motion_keys,
        num_episodes=num_episodes,
        num_failures=num_failures,
        failure_rate=failure_rate,
        bin_weights=weights,
        bin_draw_counts=draw_counts,
        realized_bin_share=realized_bin_share,
        realized_motion_share=realized_motion_share,
        recomputed_bin_prob_unweighted=recomputed,
        sampling_mass_semantics=summary.get("sampling_mass_semantics"),
        caveats=caveats,
    )


def read_checkpoint_sampler_state(path: str | Path, **kwargs: Any) -> CheckpointSamplerState:
    """Realized sampler counts from a training checkpoint's ``env_state_dict['motion_lib']``.

    Reuses ``scripts.research.dump_sampler_checkpoint_state.extract_sampler_state`` for the
    unpickling (it needs the trainer's classes importable, hence ``weights_only=False``) and
    ``build_sampler_state_summary`` for the derived quantities. Its caveats are carried
    through untouched -- notably that realized sampling mass is selected-target-bin mass
    BEFORE the ``pre_failure_sample_window`` shift, so it is not executed-start-bin mass.

    The import is deferred so that importing this module never depends on ``scripts`` being
    on the path.
    """
    from scripts.research.dump_sampler_checkpoint_state import extract_sampler_state

    return summarize_checkpoint_sampler_state(extract_sampler_state(Path(path)), **kwargs)


# ---------------------------------------------------------------------------
# 5. One row per configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcentrationReport:
    """Every concentration metric for one sampler configuration, ready to tabulate."""

    label: str
    schema_version: int
    num_bins: int
    num_motions: int
    config: dict[str, Any]
    layout: dict[str, Any]
    nominal_uniform_rate: float
    realized_uniform_mass: float
    uniform_mass_realized_over_nominal: float
    uniform_mass_residual_l1: float
    uniform_mass_caps_active: bool
    normalized_shannon_entropy: float
    effective_num_bins: float
    effective_num_bins_frac: float
    bin_prob_max: float
    bin_prob_max_over_uniform: float
    num_concentrated_bins: int
    exposure: MotionExposure
    failure_rate_cap: FailureRateCapDiagnostic
    wasted: WastedExposure | None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "schema_version": self.schema_version,
            "num_bins": self.num_bins,
            "num_motions": self.num_motions,
            "config": dict(self.config),
            "layout": dict(self.layout),
            "nominal_uniform_rate": self.nominal_uniform_rate,
            "realized_uniform_mass": self.realized_uniform_mass,
            "uniform_mass_realized_over_nominal": self.uniform_mass_realized_over_nominal,
            "uniform_mass_residual_l1": self.uniform_mass_residual_l1,
            "uniform_mass_caps_active": self.uniform_mass_caps_active,
            # Shannon, normalized by log(num_bins). NOT effective_num_bins.
            "normalized_shannon_entropy": self.normalized_shannon_entropy,
            # Renyi-2 / inverse participation ratio, the quantity SONIC logs. NOT Shannon.
            "effective_num_bins": self.effective_num_bins,
            "effective_num_bins_frac": self.effective_num_bins_frac,
            "bin_prob_max": self.bin_prob_max,
            "bin_prob_max_over_uniform": self.bin_prob_max_over_uniform,
            "num_concentrated_bins": self.num_concentrated_bins,
            "exposure": self.exposure.to_dict(),
            "failure_rate_cap": self.failure_rate_cap.to_dict(),
        }
        payload["wasted_exposure"] = None if self.wasted is None else self.wasted.to_dict()
        return payload


def concentration_report(
    layout: BinLayout,
    *,
    failure_rates: ArrayLike | None = None,
    num_failures: ArrayLike | None = None,
    num_episodes: ArrayLike | None = None,
    config: SamplerConfig = SamplerConfig(),
    active_bins: ArrayLike | None = None,
    motion_keys: Sequence[str] | None = None,
    infeasible_keys: Iterable[str] | None = None,
    strict_keys: bool = True,
    label: str = "",
) -> ConcentrationReport:
    """Measure one sampler configuration end to end.

    Runs the shipped sampler once for the headline distribution and a small number of extra
    times for the decompositions (uniform-mass fit, cap-inertness check). All of them go
    through :func:`sampling_distribution`, so every number traces back to
    ``MotionLibBase.sync_and_compute_adaptive_sampling``.
    """
    kwargs = {
        "failure_rates": failure_rates,
        "num_failures": num_failures,
        "num_episodes": num_episodes,
        "active_bins": active_bins,
    }
    prob = sampling_distribution(layout, config=config, **kwargs)
    active = _resolve_active_bins(layout, active_bins)
    exposure = per_motion_exposure(
        prob,
        layout.motion_index[active],
        motion_keys=motion_keys,
        num_motions=layout.num_motions,
    )
    decomposition = uniform_mass_decomposition(layout, config=config, **kwargs)
    cap = failure_rate_cap_diagnostic(layout, config=config, **kwargs)
    wasted = (
        None
        if infeasible_keys is None
        else wasted_exposure(exposure, infeasible_keys, strict=strict_keys)
    )

    uniform_bin_prob = 1.0 / prob.size
    return ConcentrationReport(
        label=label,
        schema_version=SAMPLER_DIAGNOSTICS_SCHEMA_VERSION,
        num_bins=int(prob.size),
        num_motions=layout.num_motions,
        config=config.to_dict(),
        layout=layout.to_dict(),
        nominal_uniform_rate=decomposition.nominal_rate,
        realized_uniform_mass=decomposition.realized_mass,
        uniform_mass_realized_over_nominal=(
            decomposition.realized_mass / decomposition.nominal_rate
            if decomposition.nominal_rate > 0
            else math.nan
        ),
        uniform_mass_residual_l1=decomposition.residual_l1,
        uniform_mass_caps_active=decomposition.caps_active,
        normalized_shannon_entropy=normalized_shannon_entropy(prob),
        effective_num_bins=effective_num_bins(prob),
        effective_num_bins_frac=effective_num_bins(prob) / prob.size,
        bin_prob_max=float(prob.max()),
        bin_prob_max_over_uniform=float(prob.max() / uniform_bin_prob),
        # Matches manager_env_wrapper.py:1027-1030's fixed 10x concentration marker.
        num_concentrated_bins=int((prob > 10.0 * uniform_bin_prob).sum()),
        exposure=exposure,
        failure_rate_cap=cap,
        wasted=wasted,
    )
