# Trajectory-Conditioned Critical Distribution

**Status:** E10b verifies multi-obstacle composition; `q_LFH_conditional_v1` passes a four-source
mechanism check and a MuJoCo cross-render; E8, fresh-motion calibration, and non-overhead support
remain open.

## Review conclusion

SweepCF should not learn a natural-world obstacle density from this corpus; that distribution is
not observed. It should learn a **designed proposal distribution** over obstacles that are feasible
for an exact executed nominal/adapted motion pair. Geometry defines support, deterministic
archetypes provide appearance, and physics alone supplies verdicts.

The final audit corrected two SweepCF-DCS v1 errors. Probe/refused-variant episodes could fill
causal targets, and 48/168 configured targets paired crouch with impossible shoulder binders.
SweepCF-DCS v2 admits only canonical cells from exact verified variants and has 120 coupling-valid
targets. Replaying the immutable CAL3 search restores 20 proposals across three sources.

## Representation

At route progress `u*`, use the executed route frame
`T(u*) = [tangent, lateral, vertical]`. Define a critical atom as:

`(source, operator, u*, n_route, keypoint, extent_tlv, mu_nom, mu_edit, xi, archetype, context)`.

`n_world = T(u*) n_route` allows the representation to address every face direction without
pretending every direction is supported. `extent_tlv` is the finite tangent/lateral/vertical face
extent. `P_feas` contains an atom only when an accepted executed nominal/adapted pair separates
along `n_route`; the current verified critical support is overhead only. Lateral context faces do
not make lateral critical support, and floor/oblique faces remain unsupported.

For exact executed reaches, `L = R_edit + delta_clear`,
`U = R_nominal - delta_strike`, and `xi = (coordinate - L) / (U - L)`. `xi` normalizes difficulty
within a motion-pair window; absolute height remains an embodiment/scene coordinate. The current
18.044 mm E1a value is applied on both sides as a conservative empirical engineering margin. It is
not a calibrated confidence interval.

`critical_support.py` and `render_critical_support.py` emit deterministic `P_feas`. Candidate
weights give each source equal mass, so a dense station/depth grid cannot dominate. DCS novelty is
a secondary ranking term. A later `q_LFH` may use maximum-entropy or hierarchical density fitting,
but never predicts or replaces physics outcomes.

For multi-obstacle scenes, sample one binding atom from `q_LFH` and context geometry from a separate
keep-out-conditioned proposal `q_ctx(G_ctx | trajectory, binding)`. Context variants receive no
causal label and no additional source mass. A future multi-binding scene requires incremental
ablation evidence for every binding obstacle; visual clutter cannot silently become a second cause.

## Fresh-motion and multi-obstacle verification

E9a sampled a new curved walking motion from Kimodo. Nominal accepted in empty-scene Isaac physics,
but its 80 mm crouch twin rejected without external contact: endpoint error was 0.376 m and p95 path
error 0.410 m. The transaction correctly stopped before obstacle construction. This establishes the
required execution order: generate → adapt → empty-scene physics → feasible support → scene →
physics. A generated reference that passes CPU checks is not yet a valid LFH source.

E10b held a verified source and simulator seed fixed, added four route-relative context obstacles
around one binding hanging panel, and reproduced accepted/accepted/rejected/accepted. The CPU
context clearance was 288.78 mm; nominal-hard contact remained uniquely on the binding panel before
drift, and all intended-clear cells had zero external contact. E10 at a different seed failed both
adapted cells without contact, so context invariance remains seed- and source-conditioned rather
than a general theorem.

## CAL3 support census

Seventeen trials retain at least 20 mm after symmetric margins. The initial deterministic
widest-per-source baseline selected:

| source | progress | depth | engineering interval | hard | easy |
|---|---:|---:|---:|---:|---:|
| `lfh_086_crouch` | 0.543890 | 0.30 m | 1.239224–1.275396 m | 1.257310 m | 1.343440 m |
| `lfh_089_crouch` | 0.473283 | 0.10 m | 1.244572–1.278107 m | 1.261340 m | 1.346151 m |
| `lfh_090_crouch` | 0.475647 | 0.10 m | 1.237501–1.263515 m | 1.250508 m | 1.331559 m |

The machine-readable evidence is `docs/hallucination/critical_support_cal3.json`.

The census is support, not a ranking oracle. E6 later showed why: source 086's widest atom used a
0.30 m route-aligned face and failed its adapted-easy context gate without contact, while the same
station with a 0.10 m face passed. A useful proposal distribution must retain route phase and
finite exposure rather than collapse each source to its largest raw window.

## E6 and E6c result

E6 spent 10 rollouts and 0.148 contended GPU-h. Sources 089 and 090 reproduced complete canonical
2x2 patterns with uniquely attributed `torso_link` contact on the binding plank before drift,
confirming the registered primary at exactly 2/3. Source 086 was refused before hard because its
adapted-easy 0.30 m exposure crossed the reference-drift threshold without external contact.

E6c was registered as an adaptive ablation, not a retroactive retry. It held source, motions,
operator, station, archetype, target, and margin rule fixed while shortening exposure from 0.30 m
to 0.10 m. All four cells then matched the canonical pattern. Nominal-hard contact was uniquely
attributed to `torso_link` on the binding plank at frame 124, before drift at frame 133; all clear
cells had zero external contact. The four-rollout ablation cost 0.056 contended GPU-h. This is one
paired intervention, not a population effect estimate, but it establishes finite exposure as a
necessary design variable for this source. See `REPORT_E6.md` and `REPORT_E6C.md`.

## E7/E7c transfer result and `q_LFH_v1`

E7 spent 24 rollouts and 0.340 contended GPU-h. Five of six `door_lintel`/`ibeam` transfers passed
the full causal gate across sources 086, 089, and 090. All outcome labels matched, but source 086's
door-lintel contact followed reference drift (frame 122 versus 95), so that variant was correctly
refused. E7c held source, station, exposure, coordinates, motions, and margins fixed and changed
only the archetype to `hanging_panel`; all four cells verified in 0.057 GPU-h, with nominal-hard
binding contact at frame 120 before drift at 135.

`q_lfh_v1.json` now assigns equal 0.25 mass to four independent trajectory sources, balances
verified archetypes within source, and splits each source-archetype atom equally between easy and
hard roles. It contains 12 physics-verified source-archetype pairs. It is an evidence-backed seed
distribution, not a fitted natural density or verdict model.

`critical_distribution.py` turns that evidence into
`q_LFH(archetype, xi | trajectory-local support, critical condition)`. It source-balances before
applying an interpretable kernel over route phase, finite exposure, engineering width,
across-route extent, and operator; a 10% exploration component remains explicit.

**The conditioning does not yet earn its place (corrected 2026-08-26).** Leave-one-source-out mean
log loss is 1.51373 against 1.60944 for a uniform prior — but **1.46416 for the same sampler with
the kernel replaced by a constant**, so feature-blind archetype counting beats the fitted model and
trajectory conditioning costs 0.0496 nats. Top-3 recall of 11/12 and 400/400 in-support sampling
are both true by construction (two archetypes are verified for every source and tie at 0.32; the
coordinate sampler maps a bounded quantile affinely onto the interval the check tests). Details and
the counting baseline are in `REPORT_Q_LFH_CONDITIONAL_MUJOCO.md` and
`REPORT_AUDIT_2026-08-26.md`.

The diagnosis is a target problem, not a fitting problem: the executed pair determines *where* a
face must be (closed form, §2.4) and does not determine *what the face looks like*. Until a
learning target is chosen that the trajectory actually constrains, source-balanced counting is the
honest proposal distribution over archetypes.

The audit found a stricter readiness condition: three archetypes per source does not create a
held-out source × archetype experiment unless the archetypes are common across sources. Only
`shelf_plank` and `ibeam` currently form a complete four-source cross. E8 should therefore transfer
`hanging_panel` to `cf_005_056`, 089, and 090, with easy/context recertification before hard cells.
When already verified archetypes are excluded, the conditional mechanism ranks `hanging_panel`
first for all three missing crosses (0.95 proposal probability versus 0.05 exploration mass on
`hvac_duct`). That ranking prioritizes E8 but is not physics evidence.

## MuJoCo visual cross-check

The E10b hard USDA scene and recorded nominal/adapted SONIC trajectories now cross-render in
MuJoCo. The bridge compiles the binding panel and four context cubes with at most `2.22e-16 m`
centre error and zero full-size error, converts the declared IsaacLab joint order to the audited
MuJoCo order, and replays `wxyz` states kinematically. The labeled side-by-side video is
`docs/source/_static/lfh_progress/e10b-mujoco-cross-render.mp4`.

The nominal view carries the recorded Isaac contact/rejection verdict and the adapted crouch carries
the recorded Isaac accepted verdict. MuJoCo is a geometry/rendering cross-check only; no MuJoCo
integration step is taken and it supplies no physics label.

## Learning-value experiment

After E8 yields at least three common verified archetypes across all four sources, compare four
equal-physics/equal-training-budget samplers:

1. source-balanced critical support (`q_LFH`),
2. uniform placement over feasible support,
3. current DCS-ranked feasible placement, and
4. visual-only variants at fixed critical geometry.

Use held-out source × held-out archetype splits, per-source weights, fixed optimizer/render budget,
and scene-visible, scene-hidden, and wrong-scene controls. Primary outcomes are unsafe choice,
unnecessary adaptation, counterfactual choice accuracy, and realized success minus adaptation
cost. This—not scene count or DCS occupancy—tests whether the learned critical distribution helps
future robot learning.
