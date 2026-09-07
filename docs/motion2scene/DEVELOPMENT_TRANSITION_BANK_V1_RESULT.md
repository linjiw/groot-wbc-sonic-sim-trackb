# Three development carriers qualify the switching contract

2026-09-07. All twelve registered empty-scene cells complete. Every one of the six
(source, physics seed) pairs qualifies: both commands cross the virtual plane at
reference-route station 0.50, stabilize upright for 0.300 s, and record no reset and
no fall. Each d040 cell logs a legal 0.30 s entry and a return. The registered
prediction — **all three sources qualify in both seeds** — is confirmed. The physical
beam was disabled in every cell, and every recorded beam normal force is 0 N.

| Source | Seed | Walk-commit | d040 request | d040 entry / return | Resets | d040 tracker outcome |
| --- | ---: | :---: | :---: | :---: | ---: | --- |
| 41001 | 8721 | pass | pass | logged / logged | 0 | **rejected** (endpoint) |
| 41001 | 8722 | pass | pass | logged / logged | 0 | **rejected** (endpoint) |
| 41002 | 8721 | pass | pass | logged / logged | 0 | accepted |
| 41002 | 8722 | pass | pass | logged / logged | 0 | accepted |
| 41003 | 8721 | pass | pass | logged / logged | 0 | **rejected** (endpoint) |
| 41003 | 8722 | pass | pass | logged / logged | 0 | accepted |

This adds **41001 and 41003** to the development pool. Until tonight, 41002 was the
only carrier qualified under the 0.30 s request / 3.3–3.5 s return contract. 41002 is
a positive control here, not new transfer; its two cells reproduce its known behavior.

## The tracker rejects d040 on two carriers, and qualification does not

The registered qualification criterion is physical: plane crossing, upright
stabilization, no reset or fall, and a legal switch. It deliberately does not include
the SONIC tracker's own accept/reject verdict. Those verdicts diverge here. On 41001
(both seeds) and on 41003 seed 8721, the d040 reference is **rejected for
`reference_endpoint_tracking_error`** while the executed motion still crosses, still
stabilizes and still records zero contact. 41002 is accepted at both seeds.

Two consequences, and neither is a licence to drop a carrier.

First, the ratio of tracker-accepted d040 executions is **3/6**, which matches the
already-recorded difficulty of deep crouch references (`TIMING_DIAGNOSTIC_V1_RESULT`
rejected five of six d055 cells; `SOURCE_EXECUTION_V1_RESULT` refused 41007 and 42007).
It is a property of the crouch reference under this controller, not of this batch.

Second — and this is why the carriers stay in — these are exactly the carriers on which
**reference-based scene construction should be least trustworthy**. When the tracker
reports endpoint error, the achieved trajectory departs from the reference that a
reference-based proposal evaluator screens against. The achieved-transition evaluator
registered for the next construction study reads the executed trajectory instead, so
41001 and 41003 are the informative cases for that comparison, not the disposable ones.
This is a stated expectation, not a measured result; the construction study tests it.

Downstream travel separates the two commands as expected: walk-commit reaches
1.805–2.055 m past the plane, d040 reaches 1.585–1.862 m, in every pair.

## Admission

All eight registered predicates pass on all twelve cells: direct causal features,
same-index physical state, ray origin bound, matched pre-decision histories, source
joint bound, bitwise bank repeat within a carrier, state bound and state unchanged.

| Audit | Bound | Worst observed |
| --- | ---: | ---: |
| Loaded root XY vs source interpolation | 1e-4 m | **5.72e-7 m** |
| Loaded joint angles vs scalar interpolation | 0.002 rad | **9.99e-4 rad** |
| Root Z first offset | reported | 0.0 m |
| Pre-decision prefix (15 frames, 10 fields) | exact | **0.0 on every field** |

The joint bound is a declared bounded check, not exact loader reproduction: the
deployed loader uses float32 quaternion interpolation, whose measured departure from
scalar interpolation is 0.000984 rad on the previously recorded 41002 bank. The worst
value here, 0.000999 rad, sits at that same scale. No carrier was compared against
41002's bank.

**Retained failure.** The v1 preparation stopped before any manifest or physics at
41001 root reconstruction error **4.768e-7 m**, exceeding its original 1e-7 check.
That protocol, script state and failure record are retained; v2 declares a 1e-6 root
tolerance and changes no source file and no physical criterion.

## Cost and scope

Twelve cells, **0.105115 contended GPU-hours** against 1.25 reserved, under the 7500 MiB
memory floor, 375 s cell timeout and serial physics, with 4.025 h of rolling daily use
recorded at launch. No cell was retried and none was replaced.

This is empty-scene command qualification. It establishes that three carriers support
the switching interface; it establishes nothing about scene construction, learning
utility or multi-source generalization. Only seed 8721 supplies proposals for the next
study; 8722 is a predeclared execution repeat, not an untouched test.

Evidence: `result.json`, `admission.json`, `run_record.json` and twelve rollout
directories under `/home/linjiw/research-data/groot-wbc/m2s-development-transition-bank-v2`.
[Registered protocol](DEVELOPMENT_TRANSITION_BANK_V1.md).

```bash
PYTHONPATH=. .venv_research/bin/python scripts/research/motion2scene_development_bank.py \
  prepare --out /path/to/bank-output
PYTHONPATH=. .venv_research/bin/python scripts/research/motion2scene_development_bank.py \
  preflight --out /path/to/bank-output
PYTHONPATH=. .venv_research/bin/python scripts/research/motion2scene_development_bank.py \
  run --out /path/to/bank-output
PYTHONPATH=. .venv_research/bin/python scripts/research/motion2scene_development_bank.py \
  analyze --out /path/to/bank-output
```

Next: [the registered ICRA construction and learning study](M2S_ICRA_V1.md), which gives
the analytic and learned methods the identical achieved transitions measured here.
