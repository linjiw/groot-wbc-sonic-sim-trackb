# LACE termination-trace autoresearch

Goal: close the scientific atlas multi-hot blocker without re-evaluating a
termination predicate or changing Isaac Lab's public done-buffer behavior.

Acceptance metric: one installed-manager compute must call every active term
exactly once, retain every simultaneous raw boolean cause in declared order,
reproduce the pinned manager's legacy timeout/terminated/last-winner buffers,
and fail closed on source, private-API, shape, freshness, or union drift. The
recorder must remain disabled and mutation-free by default.

Outcome: keep. An opt-in, per-instance, version-pinned compute wrapper now taps
each term's single evaluation into an ordered matrix while running the installed
Isaac Lab 2.3.2 aggregation logic. Each post-step trace is generation- and
`common_step_counter`-bound, checked against all three environment done buffers,
and exported with canonical manager-source, wrapper-source, and resolved-term
configuration digests. Scientific atlas ingestion now requires and validates the
per-episode matrix instead of trusting an instrument-level availability flag.
The first live launch exposed Isaac's manager-construction order; installation is
therefore deferred from `RecorderTerm.__init__` to `record_pre_reset`, with an
idempotent integrity check in `record_post_reset` and a mandatory `record_pre_step`
safeguard for callers that skip the recommended initial reset.
Atlas-probe assemblers also quiesce each environment after its one frozen row
completes, so Isaac's automatic reset of an early-done motion cannot reopen the
same rollout while longer vectorized motions are still running.

The lifecycle correction itself did not rerun the simulator or GPU. The broad
CPU gate used the repository's `.venv_sim`; an earlier system-Python attempt
exposed unrelated old NumPy/SciPy API incompatibilities and was not used as the
decision metric.
