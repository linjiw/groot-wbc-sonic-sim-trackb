"""IsaacLab adapter: the frontier teacher as a curriculum/command term for
ManagerBasedRLEnv workflows (massively parallel GPU RL, e.g. GR00T/GEAR-SONIC
humanoid tracking, locomotion terrains, manipulation goal spaces).

IsaacLab's native pattern (verified against isaaclab.managers and a production
fork, gear_sonic/envs/manager_env/mdp/curriculum.py):

  - a *curriculum term* is a function `(env, env_ids, ...) -> tensor` invoked
    by the CurriculumManager on episode resets;
  - task assignment lives in the *command manager* (which reference motion /
    goal each env tracks);
  - success/failure lives in the *termination manager* (`reset_terminated`).

frontier_rl maps onto this WITHOUT the episodic-group loop of FrontierTrainer:
in a 4096-env sim there are no discrete "groups"; instead every reset event is
one Bernoulli observation for its env's task bin.  The teacher runs in
aggregate:

  observe:  on reset, (bin, terminated-early?) pairs stream into the posterior
  sample:   resampled envs draw their next bin from the teacher distribution

This file has NO isaaclab import at module level — the term functions receive
`env` duck-typed, so the module is unit-testable on CPU (see
test_framework.py::test_isaaclab_adapter) and imports cleanly in a sim-less
venv.  Dense-reward PPO note (per SONIC_RESPONSE.md): with hazard-style
per-reset evidence and no natural group size, `utility="learnability"` is the
recommended default; advmass needs an explicit N with visits-per-recompute
semantics.

Usage sketch inside an IsaacLab task cfg:

    from frontier_rl.adapters.isaaclab_curriculum import FrontierBinTeacher

    teacher = FrontierBinTeacher(n_bins=len(bins), utility="learnability",
                                 decay_half_life=2048)   # episode-equivalents

    # 1. termination hook (call each step or on resets):
    #    teacher.observe_resets(bin_ids=env_bins[reset_ids],
    #                           failed=env.termination_manager.terminated[reset_ids])
    # 2. command/reset hook (when assigning new tasks to reset envs):
    #    new_bins = teacher.sample_bins(len(reset_ids))
    # 3. optional telemetry:  logger.log(teacher.metrics(), step)
"""

from __future__ import annotations

import numpy as np


class FrontierBinTeacher:
    """Vectorized frontier teacher for massively parallel reset streams.

    Differences from frontier_rl.teacher.FrontierTeacher, each forced by the
    parallel-sim setting and consistent with SONIC_RESPONSE.md:

    - evidence-scaled decay: a half-life in *episode-equivalents* rather than
      per-observe decay, so behavior is invariant to env count (Q4);
    - deterministic optimism (mean + k*std) by default instead of Thompson —
      reproducibility guardrails in sim RL favor determinism (Q3); pass
      thompson=True to restore sampling;
    - utility="learnability" default (no natural N in a reset stream, Q2);
    - a max_prob tripwire instead of a shaping cap (Q9).
    """

    def __init__(self, n_bins: int, *, utility: str = "learnability",
                 advmass_n: int = 16, decay_half_life: float = 2048.0,
                 floor: float = 0.1, gamma: float = 1.0,
                 optimism_k: float = 1.0, thompson: bool = False,
                 max_prob: float | None = None, seed: int = 0):
        assert utility in ("learnability", "advmass")
        self.n_bins = n_bins
        self.utility_kind = utility
        self.advmass_n = advmass_n
        self.half_life = decay_half_life
        self.floor = floor
        self.gamma = gamma
        self.optimism_k = optimism_k
        self.thompson = thompson
        self.max_prob = max_prob
        self.rng = np.random.default_rng(seed)
        self.succ = np.zeros(n_bins)
        self.fail = np.zeros(n_bins)
        self._probs = np.full(n_bins, 1.0 / n_bins)
        self._dirty = True

    # -- evidence (vectorized) ---------------------------------------------
    def observe_resets(self, bin_ids, failed) -> None:
        """Stream one batch of reset events.

        Args:
          bin_ids: int array/tensor (M,) — the bin each resetting env was on.
          failed:  bool array/tensor (M,) — early-termination flag per env.

        Accepts torch tensors (any device) or numpy — IsaacLab curriculum
        terms hand us CUDA tensors; the teacher's own state is tiny, so a
        single small D2H copy per reset batch is the cheap direction.
        """
        if hasattr(bin_ids, "cpu"):
            bin_ids = bin_ids.detach().cpu().numpy()
        if hasattr(failed, "cpu"):
            failed = failed.detach().cpu().numpy()
        bin_ids = np.asarray(bin_ids, dtype=int)
        failed = np.asarray(failed, dtype=bool)
        n_events = len(bin_ids)
        if n_events == 0:
            return
        # evidence-scaled decay: half-life in episode-equivalents
        decay = 0.5 ** (n_events / self.half_life)
        self.succ *= decay
        self.fail *= decay
        np.add.at(self.succ, bin_ids[~failed], 1.0)
        np.add.at(self.fail, bin_ids[failed], 1.0)
        self._dirty = True

    # -- posterior + utility -------------------------------------------------
    def _posterior(self):
        a = 1.0 + self.succ
        b = 1.0 + self.fail
        return a, b

    def _utility(self):
        a, b = self._posterior()
        if self.thompson:
            p = self.rng.beta(a, b)
        else:
            mean = a / (a + b)
            std = np.sqrt(mean * (1 - mean) / (a + b + 1))
            p = np.clip(mean + self.optimism_k * std, 1e-4, 1 - 1e-4)
        if self.utility_kind == "learnability":
            u = p * (1.0 - p)
        else:
            u = np.maximum((1.0 - (1.0 - p) ** self.advmass_n) - p, 0.0)
        return u ** self.gamma

    def _recompute(self):
        u = self._utility()
        if u.sum() <= 1e-12:
            u = np.ones(self.n_bins)
        probs = u / u.sum()
        probs = (1 - self.floor) * probs + self.floor / self.n_bins
        if self.max_prob is not None:   # tripwire, not shaping
            probs = np.minimum(probs, self.max_prob)
            probs = probs / probs.sum()
        self._probs = probs
        self._dirty = False

    # -- sampling -------------------------------------------------------------
    def sample_bins(self, m: int) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self.rng.choice(self.n_bins, size=m, p=self._probs)

    def sampling_probs(self) -> np.ndarray:
        if self._dirty:
            self._recompute()
        return self._probs.copy()

    # -- telemetry / persistence ----------------------------------------------
    def pass_rate_estimates(self) -> np.ndarray:
        a, b = self._posterior()
        return a / (a + b)

    def argmax_utility(self) -> int:
        """The frontier bin: where the (mean-posterior) utility peaks."""
        a, b = self._posterior()
        p = a / (a + b)
        if self.utility_kind == "learnability":
            u = p * (1.0 - p)
        else:
            u = np.maximum((1.0 - (1.0 - p) ** self.advmass_n) - p, 0.0)
        return int(np.argmax(u))

    def dead_fraction(self, threshold: float = 0.05) -> float:
        """Fraction of *seen* bins the posterior currently rates unlearnable."""
        p = self.pass_rate_estimates()
        seen = (self.succ + self.fail) > 1.0
        if not seen.any():
            return 0.0
        return float((p[seen] < threshold).mean())

    def mastered_fraction(self, threshold: float = 0.9) -> float:
        p = self.pass_rate_estimates()
        seen = (self.succ + self.fail) > 1.0
        if not seen.any():
            return 0.0
        return float((p[seen] > threshold).mean())

    def metrics(self) -> dict:
        seen = (self.succ + self.fail) > 1.0
        return {"teacher/effective_bins": float(1.0 / (self.sampling_probs() ** 2).sum()),
                "teacher/seen_frac": float(seen.mean()),
                "teacher/frontier_bin": float(self.argmax_utility()),
                "teacher/frac_dead": self.dead_fraction(),
                "teacher/frac_mastered": self.mastered_fraction()}

    def state_dict(self) -> dict:
        return {"succ": self.succ.copy(), "fail": self.fail.copy()}

    def load_state_dict(self, state: dict) -> None:
        self.succ = np.asarray(state["succ"], dtype=float)
        self.fail = np.asarray(state["fail"], dtype=float)
        self._dirty = True


# --------------------------------------------------------------------------
# IsaacLab curriculum term (the §9.2 integration from the infra guide)
# --------------------------------------------------------------------------
class FrontierTerrainTeacherTerm:
    """A stateful curriculum term for ManagerBasedRLEnv, terrain-level axis.

    Drop-in for the stock ``terrain_levels_vel``: observe the ending episodes'
    outcomes, Thompson/optimism-sample next-episode terrain levels from the
    frontier teacher, write them to ``terrain.terrain_levels`` + env origins.
    Runs inside ``CurriculumManager.compute(env_ids)`` — i.e. in
    ``_reset_idx`` BEFORE ``scene.reset``, so written origins take effect for
    the episode about to start (the contract ``terrain_levels_vel`` uses).

    IsaacLab-independent by construction: `env` is duck-typed
    (needs .scene.terrain{.terrain_levels,.max_terrain_level,.env_origins,
    .terrain_origins,.terrain_types} and .termination_manager.time_outs), so
    the class is unit-testable on CPU with a stub env.  To register in a task
    cfg, subclass isaaclab.managers.ManagerTermBase and delegate, or use it
    directly as ``CurrTerm(func=FrontierTerrainTeacherTerm(cfg_dict), ...)``
    if your fork accepts callables with state (most do).

    Success signal: ``time_outs`` (survived to timeout) by default — swap in a
    task predicate via ``success_fn(env, env_ids) -> bool tensor`` for tasks
    where survival saturates (see the infra guide §9.3).
    """

    def __init__(self, n_bins: int = None, *, utility: str = "learnability",
                 decay_half_life: float = 2048.0, floor: float = 0.1,
                 optimism_k: float = 1.0, thompson: bool = False,
                 success_fn=None, seed: int = 0):
        self._pending = dict(utility=utility, decay_half_life=decay_half_life,
                             floor=floor, optimism_k=optimism_k,
                             thompson=thompson, seed=seed)
        self.teacher = (FrontierBinTeacher(n_bins, **{
            "utility": utility, "decay_half_life": decay_half_life,
            "floor": floor, "optimism_k": optimism_k,
            "thompson": thompson, "seed": seed}) if n_bins else None)
        self.success_fn = success_fn

    def _ensure_teacher(self, env):
        if self.teacher is None:
            n_bins = int(env.scene.terrain.max_terrain_level)
            kw = self._pending
            self.teacher = FrontierBinTeacher(
                n_bins, utility=kw["utility"],
                decay_half_life=kw["decay_half_life"], floor=kw["floor"],
                optimism_k=kw["optimism_k"], thompson=kw["thompson"],
                seed=kw["seed"])

    def __call__(self, env, env_ids, uniform_floor: float = None):
        import torch
        self._ensure_teacher(env)
        if uniform_floor is not None:
            self.teacher.floor = uniform_floor
        terrain = env.scene.terrain
        # 1) OBSERVE the episodes that just ended
        bins = terrain.terrain_levels[env_ids]
        if self.success_fn is not None:
            success = self.success_fn(env, env_ids)
        else:
            success = env.termination_manager.time_outs[env_ids]
        succ = success if not hasattr(success, "cpu") else success
        self.teacher.observe_resets(bins, ~succ if hasattr(succ, "__invert__")
                                    else np.logical_not(succ))
        # 2) SAMPLE next-episode difficulty for exactly these envs
        new_bins = torch.as_tensor(self.teacher.sample_bins(len(env_ids)),
                                   device=terrain.terrain_levels.device,
                                   dtype=terrain.terrain_levels.dtype)
        terrain.terrain_levels[env_ids] = new_bins.clamp(
            0, int(terrain.max_terrain_level) - 1)
        terrain.env_origins[env_ids] = terrain.terrain_origins[
            terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
        # 3) TELEMETRY -> Curriculum/<term>/* in TensorBoard
        m = self.teacher.metrics()
        return {"mean_bin": terrain.terrain_levels.float().mean(),
                "frontier_bin": m["teacher/frontier_bin"],
                "dead_frac": m["teacher/frac_dead"],
                "mastered_frac": m["teacher/frac_mastered"],
                "effective_bins": m["teacher/effective_bins"]}

    # persistence (call from your checkpoint hooks)
    def state_dict(self):
        return self.teacher.state_dict() if self.teacher else {}

    def load_state_dict(self, state):
        if self.teacher and state:
            self.teacher.load_state_dict(state)


def make_curriculum_term(teacher: FrontierBinTeacher, bin_of_env_fn,
                         command_name: str = None):
    """Minimal function-style wrapper (kept for command-manager-side axes,
    where task ASSIGNMENT lives in a custom command term and this term only
    feeds evidence).  See FrontierTerrainTeacherTerm for the full pattern."""
    def frontier_curriculum(env, env_ids, **kwargs):
        import torch
        ids = env_ids if not hasattr(env_ids, "cpu") else env_ids.cpu().numpy()
        bins = bin_of_env_fn(env, ids)
        failed = env.termination_manager.terminated[env_ids]
        teacher.observe_resets(bins, failed)
        return torch.tensor(teacher.metrics()["teacher/effective_bins"])
    return frontier_curriculum
