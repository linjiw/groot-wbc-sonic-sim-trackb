#!/usr/bin/env python3
"""Roll out a deployable skill in its own family's scenes, and score the result the same way.

`build_deployable_skill.py` retimes an adapted clip that cleared its obstacle and was rejected for
arriving short. Whether that worked is not a property of the clip -- it is a rollout, and the answer
has to come from the same gate, the same scenes and the same scorer that rejected the original, or
the comparison is between two different measurements rather than two clips.

So this re-rolls **only the two adapted cells**. The nominal cells are reused from the causal
family, and that is not a shortcut: the retiming touches the adapted clip alone, so a re-rolled
nominal would differ from the recorded one only by simulator noise, and spending two GPU rollouts to
introduce that noise into the comparison is the opposite of what the 2x2 is for. The reused cells
are linked rather than copied so their provenance still points at the run that produced them.

The card here is shared, and a rollout started against a full one does not run slowly, it dies --
so capacity is waited for rather than assumed, per `gpu_capacity`.

Usage::

    python scripts/research/run_deployable_family.py \\
        --skill /data/.../wsA/deployable/n_013_ceiling_overhead_left_deployable.pkl \\
        --causal /data/.../wsA/families/n_013_ceiling_overhead_left \\
        --scene-easy gf_013_ceiling_overhead_left_easy \\
        --scene-hard gf_013_ceiling_overhead_left_hard \\
        --out /data/.../wsA/deployable_family/n_013_ceiling_overhead_left
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.gpu_capacity import wait_for_gpu  # noqa: E402

#: The task string the causal family was captured with. It is bound into the runtime manifest and
#: cross-checked at export, so it has to match or the two families are not comparable.
TASK = "walk forward through the room"

#: Cells this script produces, against the cells it reuses.
REROLLED = ("adapted_easy", "adapted_hard")
REUSED = ("nominal_easy", "nominal_hard")


def rollout(scene: str, motion: Path, out: Path, *, checkpoint: Path | None = None) -> bool:
    """One physics rollout. Success is the explicit marker, never the exit status."""
    out.mkdir(parents=True, exist_ok=True)
    log = out / "driver.log"
    wait_for_gpu()
    command = [
        str(REPO_ROOT / "scripts/research/run_kimodo_sonic_rollout.sh"),
        "--scene",
        scene,
        "--motion",
        str(motion),
        "--out",
        str(out),
        "--max-steps",
        "auto",
        "--task",
        TASK,
    ]
    if checkpoint is not None:
        command += ["--checkpoint", str(checkpoint)]
    with log.open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    text = log.read_text(encoding="utf-8", errors="replace")
    return "PASS" in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True, help="the retimed motion_lib PKL")
    parser.add_argument("--causal", type=Path, required=True, help="the family it was retimed from")
    parser.add_argument("--scene-easy", required=True)
    parser.add_argument("--scene-hard", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-only", action="store_true", help="skip the rollouts")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Reuse the nominal cells by symlink, so the scorer sees a complete 2x2 and the provenance of
    # those two cells still resolves to the run that actually produced them.
    for cell in REUSED:
        source = (args.causal / cell).resolve()
        if not source.exists():
            print(f"missing reused cell {source}", file=sys.stderr)
            return 2
        link = args.out / cell
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(source, target_is_directory=True)
        print(f"reused  {cell:14s} -> {source}")

    if not args.score_only:
        for cell, scene in zip(REROLLED, (args.scene_easy, args.scene_hard)):
            print(f"\nrolling {cell:14s} in {scene}")
            ok = rollout(scene, args.skill, args.out / cell, checkpoint=args.checkpoint)
            print(f"  {'PASS' if ok else 'NO MARKER'} -- {(args.out / cell / 'driver.log')}")
            if not ok:
                print(
                    "  a rollout without the success marker is not a rejected cell, it is an "
                    "unevaluable one; stopping rather than scoring it as evidence",
                    file=sys.stderr,
                )
                return 3

    root = args.out.parent
    scores = args.out / "family_scores.json"
    print(f"\nscoring {root}")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/research/score_family_batch.py"),
            str(root),
            "--json",
            str(scores),
        ],
        check=False,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )
    if scores.exists():
        print(json.dumps(json.loads(scores.read_text()), indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
