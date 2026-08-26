# E4 SweepCF-DCS Before/After Report

E4 adds the two E2-verified archetype variants to an isolated index view; E3 contributes no rows
because neither tail branch has yet verified a family. This final audit uses SweepCF-DCS v2:
probe/refused-variant rows cannot fill targets, and 48 impossible crouch/shoulder targets are absent.
The result remains **more verified visual variants but no new causal source or DCS bin**.

| metric | Phase-1 baseline | after E2/E3 | change |
|---|---:|---:|---:|
| indexed episodes | 68 | 76 | +8 |
| graded episodes | 60 | 68 | +8 |
| verified variants | 3 | 5 | +2 |
| independent causal families | 2 | 2 | +0 |
| occupied target bins | 2/120 | 3/120 | +1 |
| target-bin coverage | 1.67% | 2.50% | 0.83% |

Door-lintel and I-beam add eight physics-graded episodes while remaining variants of
`cf_005_056`; they cannot advance the independent-family denominator. Their binding coordinates
and causal-family margin buckets were already occupied, so verified-family target coverage remains
3/120. The E4 index and plots live under
`docs/hallucination/e4_index/` and
`docs/hallucination/e4_coverage/`; they remain separate from the release and claim-5 inputs.
