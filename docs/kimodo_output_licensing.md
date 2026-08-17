# Kimodo output licensing — can we release the motions?

**Answer: yes.** Generated motions and every episode derived from them can be released, including
commercially. The plan flagged this as a risk to settle before Phase F because it decides whether we
release episodes, manifests-plus-regeneration-scripts, or both. We can release both.

Checked 2026-08-16 against the license shipped inside the model repo itself
(`nvidia/Kimodo-G1-RP-v1`, snapshot `3020ad8c`), not against a summary of it.

## What the license says

The model is under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Three clauses decide our case:

> - Models are commercially usable.
> - You are free to create and distribute Derivative Models.
> - NVIDIA does not claim ownership to any outputs generated using the Models or Derivative Models.

and, in the definitions:

> 1.1. "Derivative Model" means all (a) modifications to the Model, (b) works based on the Model, and
> (c) any other derivative works of the Model. **An output is not a Derivative Model.**

and:

> 2.4. […] NVIDIA claims no ownership rights in outputs. You are responsible for outputs and their
> subsequent uses.

So a generated qpos clip is an *output*, not a Derivative Model. The redistribution conditions in
§3 — ship a copy of the agreement, include a "Licensed by NVIDIA Corporation under the NVIDIA Open
Model License" notice — attach to distributing **the model weights**, which we are not doing. The
"Built on NVIDIA Cosmos" attribution in §3.2 is Cosmos-specific and does not apply to Kimodo.

The model card also names our exact use case as intended:

> The model is intended for users […] to create 3D humanoid robot motion data for their application.
> This may include demonstrations for humanoid robots or robot motions for simulations and synthetic data.

## What this means for the release

- **Release the episodes.** No obligation to withhold motions or ship them as regeneration scripts only.
- **Do not vendor the weights** into the dataset repo. Nothing needs them — the generation env pulls
  `nvidia/Kimodo-G1-RP-v1` from the Hub — and vendoring is the one act that would trigger §3's
  copy-of-agreement and notice conditions.
- **Attribute anyway.** The dataset card should name Kimodo-G1-RP-v1 with its snapshot hash as the
  motion source. That is provenance hygiene, and it is what makes the corpus regenerable by others.
- **We own the derived work.** Scenes, acceptance labels, action tokens, and the corpus itself are
  ours to license as we choose (§3.3 explicitly permits our own terms on our own contributions).
- **We carry responsibility for outputs** (§2.4). Consistent with how the corpus is already gated:
  every released episode passes physics acceptance, and failures ship quarantined with reason codes
  rather than silently dropped.

## The one thing still open

`meta-llama/Meta-Llama-3-8B-Instruct` — the LLM2Vec text encoder Kimodo conditions on — is
**gated=manual** on the Hub and carries the Llama 3 Community License, which is not the same
instrument as the NVIDIA Open Model License. It never touches the released artifacts: it encodes the
prompt, and its weights are neither redistributed nor embedded in any output. But anyone reproducing
generation must accept Llama 3's terms themselves, so the reproduction instructions must say so
rather than let a `snapshot_download` fail mysteriously at someone else's desk.

## Incidental constraint worth recording

The model card states: **maximum duration is 10 s (300 frames at 30 fps)**. This is a hard design
constraint on Phase B, not a licensing matter — single-segment motions cannot exceed 10 s, so longer
routes must come from multi-prompt stitching (`multi_prompt=True` with `num_transition_frames`), and
the plan's "longer stitched multi-segment routes" item is a stitching problem, not a sampling one.
