# M2S-ICRA execution supplement: deterministic fits and fixed comparators

Implementation supplement registered before fitting or evaluation. It does not change
M2S_ICRA_V1's construction, labels or controller. Label acquisition may yield fewer
than 24 complete groups; fitting records the shortfall instead of manufacturing labels.
All 82 eligible command assignments must finish and pass measurement admission before
four primary fits. A scientific failure remains a label; an invalid pair stops fitting.

The learned execution code is a separately named copy of the frozen learned-command
runtime. Its only functional difference is a readout adapter: .pt files call the same
frozen learned readout; the scripted comparator reads the actual packet's occupied
bit at 0.30 s; the privileged comparator reads its hash-pinned, precomputed nominal
achieved-geometry outcome choice for that scene. Both comparators use the same request,
return and legality code. No scene rule intervenes beneath a learned model. CPU tests
bind the adapter to the original readout and source transformation; simulation records
must bind actual features, source bank, direct decision, placement and paired approach.

Fit four deterministic primary models and six leave-four-assigned-group-out refits per
arm. A refit is evaluated only on physically labelled members of its withheld group;
report both missing members and zero-member folds. No refit becomes a deployed policy
or new independent dataset. Compile primary linear functions into the existing ReLU
container and require <=2e-6 probability error and identical requests over the entire
admitted corpus before generating evaluation manifests. No test outcomes select a model.

The main panel remains 432 traversal cells (12 layouts × 2 seeds × 3 carriers × 6
methods). Fixed order is layout, seed, carrier, method; batches contain all six methods
for one matched condition. Add a separately labelled background panel: absent/raised/
blocked at station 0.60, both seeds, three carriers and six methods = 108 cells.
Those controls match the previously declared control family; they are not an unseen-
background claim. Total evaluation is 540 assignments, with the 432 traversal endpoint
unchanged. The added 108 cells implement the guidance's separate unnecessary-adaptation
and blocked-refusal endpoints; they never enter traversal passage averages.

Physics remains serial, <=8 cells per batch, 7500 MiB free before each launch,
375 s timeout, and the standing rolling 8 h/day /24 h/week limits. A master file lists
assignments; it is not authorization to reserve or run the whole panel simultaneously.
No pending rollout may be replaced by an offline outcome lookup.

All six policies' recorded pre-decision states, packets, features and command histories
must match within each condition. Readouts are recomputed on recorded inputs; source
banks are checked against source CSV/PKL and each other. Beam geometry and synchronized
contact records are audited. Legal denied commands remain outcomes, not filtered trials.
A 0.30 s decision packet without the expected model hash invalidates measurement admission.
