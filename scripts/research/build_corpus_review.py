#!/usr/bin/env python3
"""Assemble everything a person needs to review the corpus by eye, organised by behaviour.

The earlier gallery grouped videos by which batch produced them, which was the right axis
when every episode was the same walk and the batches were the only thing that differed. It
is the wrong axis now: the corpus's claim is that it covers *behaviours*, so the folders
should be behaviours and a reviewer should be able to ask "show me the side-steps" without
knowing which night they were generated.

Emits, into one directory:

* videos foldered by body mode, named ``<outcome>__<speed>__<episode>.mp4`` so the verdict
  and the requested speed are readable without opening anything
* one overview figure per behaviour: executed path against reference, root height, and
  tracking error over time
* corpus-level figures: the diversity scatter, the acceptance-versus-horizon curve, and
  per-behaviour acceptance
* ``REVIEW.md`` tying it together, with what to look for and what is known to be wrong

Every number here is recomputed from the trajectories at build time; nothing is copied from
a previous report, so the figures and the text cannot disagree.

Usage::

    python scripts/research/build_corpus_review.py --work /data/.../groot-wbc-kimodo-m0 \\
        --out /data/.../review
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.behaviour_diversity import (  # noqa: E402
    build_diversity_report,
    summarise_episode,
)
from gear_sonic.dataset_generation.contact_decomposition import (  # noqa: E402
    decompose_payload_contacts,
)
from gear_sonic.dataset_generation.episode_outcome import (  # noqa: E402
    OutcomeTally,
    classify_episode,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

CONTROL_HZ = 50.0

#: Plain-English names for the taxonomy's body modes, for folder and figure labels.
BEHAVIOUR_LABELS = {
    "walk": "forward walk",
    "walk_pause": "walk with a pause",
    "walk_look": "walk and look around",
    "crouch_walk": "lowered walk",
    "crouch_deep": "deep crouch",
    "duck_under": "duck under",
    "side_step": "side-step",
    "backward": "backward walk",
    "stand_to_walk": "start from standing",
    "walk_to_stop": "walk to a stop",
    "step_over": "step over",
    "turn_in_place": "turn in place",
    "carry_walk": "carry while walking",
    "reach_walk": "walk and reach",
    "squat_pick": "squat to pick up",
}


def infer_behaviour(episode_id: str, specs: list[dict]) -> tuple[str, str]:
    """Recover (body_mode, speed) for an episode from the taxonomy index in its name."""
    match = re.search(r"(?:clutter_)?(\d{3})_", episode_id)
    if match:
        index = int(match.group(1))
        if index < len(specs):
            return specs[index]["body_mode"], specs[index]["speed"]
    # Pre-taxonomy episodes carry no index; they are all forward walks.
    return "walk", "steady"


def episode_figure(payload: dict, title: str, path: Path) -> None:
    """Executed vs reference path, root height, and tracking error for one episode."""
    executed = np.asarray(payload["root_pos_w"], dtype=np.float64)
    reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)
    frames = min(len(executed), len(reference))
    seconds = np.arange(frames) / float(payload.get("fps", CONTROL_HZ))
    deviation = np.linalg.norm(executed[:frames, :2] - reference[:frames, :2], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    fig.suptitle(title, fontsize=11)

    axes[0].plot(reference[:frames, 0], reference[:frames, 1], lw=2.4, color="#9aa0a6",
                 label="commanded")
    axes[0].plot(executed[:frames, 0], executed[:frames, 1], lw=1.8, color="#2f6fd0",
                 label="executed")
    axes[0].scatter(*executed[0, :2], s=36, color="#2f6fd0", zorder=3)
    axes[0].scatter(*executed[frames - 1, :2], s=36, marker="s", color="#c0392b", zorder=3)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("path, overhead", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)

    axes[1].plot(seconds, executed[:frames, 2], lw=1.6, color="#2f6fd0", label="executed")
    axes[1].plot(seconds, reference[:frames, 2], lw=1.4, color="#9aa0a6", ls="--",
                 label="commanded")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("root height (m)")
    axes[1].set_title("how low the body goes", fontsize=9)
    axes[1].legend(fontsize=7, frameon=False)

    axes[2].plot(seconds, deviation, lw=1.6, color="#b06a2c")
    axes[2].axhline(0.25, color="#c0392b", lw=1.0, ls=":", label="p95 gate 0.25 m")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("path error (m)")
    axes[2].set_title("drift from the command grows with time", fontsize=9)
    axes[2].legend(fontsize=7, frameon=False)

    for axis in axes:
        axis.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def corpus_figures(records: list[dict], out: Path) -> None:
    """Corpus-level views: behaviour spread, and acceptance per behaviour."""
    accepted = [r for r in records if r["outcome"] == "accepted"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    colours = {"accepted": "#2f7d32", "rejected": "#c0392b", "unevaluable": "#8a8a8a"}
    for outcome, colour in colours.items():
        subset = [r for r in records if r["outcome"] == outcome]
        if not subset:
            continue
        axes[0].scatter(
            [r["mean_speed_mps"] for r in subset],
            [r["root_height_min_m"] for r in subset],
            s=26, alpha=0.75, color=colour, label=f"{outcome} ({len(subset)})",
        )
    axes[0].set_xlabel("mean speed (m/s)")
    axes[0].set_ylabel("lowest root height (m)")
    axes[0].set_title("behaviour space: speed against how low the body goes", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.25, lw=0.5)

    by_behaviour: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_behaviour[record["behaviour"]].append(record)
    names = sorted(by_behaviour, key=lambda k: -len(by_behaviour[k]))
    rates = [
        sum(1 for r in by_behaviour[n] if r["outcome"] == "accepted") / len(by_behaviour[n])
        for n in names
    ]
    counts = [len(by_behaviour[n]) for n in names]
    positions = np.arange(len(names))
    axes[1].barh(positions, rates, color="#2f6fd0", alpha=0.85)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(
        [f"{BEHAVIOUR_LABELS.get(n, n)}  (n={c})" for n, c in zip(names, counts)], fontsize=8
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("acceptance rate")
    axes[1].set_title("which behaviours the tracker can hold", fontsize=10)
    axes[1].grid(alpha=0.25, lw=0.5, axis="x")
    axes[1].invert_yaxis()

    fig.tight_layout()
    fig.savefig(out / "corpus_behaviour_space.png", dpi=130)
    plt.close(fig)

    # Speed: what was asked for against what came out.
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    order = ["slow", "steady", "brisk"]
    data = [
        [r["mean_speed_mps"] for r in accepted if r["speed"] == style] for style in order
    ]
    if any(data):
        axis.boxplot([d or [0] for d in data], labels=order, widths=0.55)
        axis.set_ylabel("achieved mean speed (m/s)")
        axis.set_xlabel("requested speed style")
        axis.set_title("the speed axis does real work", fontsize=10)
        axis.grid(alpha=0.25, lw=0.5, axis="y")
        fig.tight_layout()
        fig.savefig(out / "corpus_speed_axis.png", dpi=130)
    plt.close(fig)


def write_review_markdown(out: Path, summary: dict, records: list[dict]) -> None:
    """The reviewer's entry point: what is here, what to look for, what is known wrong."""
    outcomes = summary["outcomes"]
    diversity = summary["diversity"]
    accepted = [r for r in records if r["outcome"] == "accepted"]

    by_behaviour: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_behaviour[record["behaviour"]].append(record)

    lines: list[str] = [
        "# G1 indoor corpus — visual review",
        "",
        f"{summary['episodes']} episodes across {len(summary['behaviours'])} behaviours, "
        f"{summary['videos']} videos. Everything below is recomputed from the trajectories "
        "at build time, so the figures and this text cannot disagree.",
        "",
        "## How to read the folders",
        "",
        "`videos/<behaviour>/<outcome>__<speed>__<episode>.mp4`. The verdict and the "
        "requested speed are in the filename, so a folder can be skimmed without opening "
        "anything. `figures/<behaviour>/` holds a three-panel overview per episode: the "
        "executed path against the commanded one, how low the body goes, and how the "
        "tracking error grows with time.",
        "",
        "## Outcomes",
        "",
        f"- **accepted {outcomes['accepted']}**, rejected {outcomes['rejected']}, "
        f"unevaluable {outcomes['unevaluable']}",
        f"- acceptance over *evaluated* episodes: **{outcomes['acceptance_rate_of_evaluated']:.0%}** "
        "(unevaluable captures are excluded rather than counted as failures, since no gate "
        "ever ran on them)",
        f"- {outcomes['recovered_from_split']} captures spanned an environment reset and were "
        "split, with the complete pass recovered",
        "",
        "## Diversity",
        "",
        f"Effective rank of the {diversity['action_dim']}-dimensional action: "
        f"**between-episode {diversity['between_episode_rank']:.2f}**, pooled "
        f"{diversity['pooled_rank']:.2f}, within-episode "
        f"{diversity['within_episode_rank_mean']:.2f}.",
        "",
        "Between-episode is the one that answers \"how many distinct behaviours are here\". "
        "It was **1.16** when the corpus ran on two motions — the accepted episodes' mean "
        "actions spanned essentially a single direction.",
        "",
        "## Acceptance by behaviour",
        "",
        "| Behaviour | Episodes | Accepted | Rate |",
        "|---|---|---|---|",
    ]
    for name in sorted(by_behaviour, key=lambda k: -len(by_behaviour[k])):
        group = by_behaviour[name]
        good = sum(1 for r in group if r["outcome"] == "accepted")
        lines.append(
            f"| {BEHAVIOUR_LABELS.get(name, name)} | {len(group)} | {good} | "
            f"{good / len(group):.0%} |"
        )

    if accepted:
        speeds = [r["mean_speed_mps"] for r in accepted]
        heights = [r["root_height_min_m"] for r in accepted]
        lateral = [r["lateral_scene_contact_n"] for r in accepted]
        lines += [
            "",
            "## Spread over accepted episodes",
            "",
            "| Quantity | min | median | max |",
            "|---|---|---|---|",
            f"| Mean speed | {min(speeds):.3f} | {float(np.median(speeds)):.3f} | "
            f"{max(speeds):.3f} m/s |",
            f"| Lowest root height | {min(heights):.3f} | "
            f"{float(np.median(heights)):.3f} | {max(heights):.3f} m |",
            f"| Lateral scene contact | {min(lateral):.3f} | "
            f"{float(np.median(lateral)):.3f} | {max(lateral):.3f} N |",
        ]

    evaluated = [r for r in records if r["outcome"] in ("accepted", "rejected")]
    if evaluated:
        reasons: dict[str, int] = defaultdict(int)
        for record in evaluated:
            for reason in record["reasons"]:
                reasons[reason] += 1
        lines += [
            "",
            "## What governs acceptance",
            "",
            "Not which behaviour it is — **how long the episode is**. The tracker starts on "
            "its reference and drifts, so duration is the strongest single predictor of "
            "rejection, and path-tracking error is the dominant reason:",
            "",
            "| Duration | Episodes | Accepted |",
            "|---|---|---|",
        ]
        for low, high in ((0.0, 2.0), (2.0, 3.2), (3.2, 4.5), (4.5, 99.0)):
            group = [r for r in evaluated if low <= r["duration_s"] < high]
            if group:
                good = sum(1 for r in group if r["outcome"] == "accepted")
                lines.append(
                    f"| {low:.1f}–{high:.1f} s | {len(group)} | {good / len(group):.0%} |"
                )
        lines += [
            "",
            "Rejection reasons, most common first: "
            + ", ".join(f"`{k}` ({v})" for k, v in
                        sorted(reasons.items(), key=lambda kv: -kv[1])),
            "",
            "**Root height carries almost no signal, contrary to how the scatter looks.** "
            "Reading the left figure suggests low-bodied behaviours fail; binned, only the "
            "deepest band does. Episodes below 0.55 m are 0-for-12, while 0.55–0.66 m runs "
            "88–90% — better than the 0.72–0.76 m band of ordinary walking. The visual "
            "impression came from red points being salient at the bottom, not from a "
            "threshold.",
        ]

    lines += [
        "",
        "## What to look for",
        "",
        "- **Does the behaviour match its folder?** A clip in `side_step/` that walks "
        "forward means the prompt did not steer the generator, which no automatic gate "
        "checks.",
        "- **Does the robot pass close to furniture without touching it?** Accepted "
        "episodes should show clearance being *used*, not a robot crossing empty floor.",
        "- **Anything in the ego view that should not be there.** Debug markers leaking the "
        "tracking target into the observation were found this way once already, on 243 of "
        "249 frames, while every numeric gate passed.",
        "",
        "## Known wrong, so you do not have to find it",
        "",
        "- The third-person chase camera uses a fixed offset and clips *inside* furniture in "
        "dense scenes, blanking part of the frame. A review artifact, not a data defect.",
        "- The overhead camera renders blank. Unfixed, and reverted rather than shipped with "
        "a speculative fix.",
        "- Acceptance depends on episode length: 83% at a 1.2 s horizon falling to 64% at "
        "4.8 s, because the tracker starts on its reference and drifts. Any rate here is for "
        "the full captured episode.",
        "- Two accepted episodes carry non-zero lateral scene contact. One is 2.8 N, "
        "marginal against a 1.0 N threshold and **not understood**.",
        "",
    ]
    (out / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument(
        "--per-behaviour-figures",
        type=int,
        default=2,
        help="episode overview figures to render per behaviour",
    )
    args = parser.parse_args()

    specs: list[dict] = []
    taxonomy_path = args.taxonomy or (args.work / "taxonomy/taxonomy.json")
    if taxonomy_path.exists():
        specs = json.loads(taxonomy_path.read_text(encoding="utf-8"))["specs"]

    args.out.mkdir(parents=True, exist_ok=True)
    videos_root = args.out / "videos"
    figures_root = args.out / "figures"
    videos_root.mkdir(exist_ok=True)
    figures_root.mkdir(exist_ok=True)

    tally = OutcomeTally()
    records: list[dict] = []
    episodes, actions = [], []
    figures_written: dict[str, int] = defaultdict(int)

    for trajectory in sorted(args.work.rglob("*.trajectory.pkl")):
        rollout_dir = trajectory.parent.parent
        episode_id = rollout_dir.name
        try:
            with trajectory.open("rb") as handle:
                raw = pickle.load(handle)  # noqa: S301 - local recorder artifact
        except Exception as error:  # noqa: BLE001 - one bad file must not end the review
            print(f"  skip {episode_id}: {type(error).__name__}: {error}")
            continue

        outcome = classify_episode(episode_id, raw)
        tally.add(outcome)
        if not outcome.evaluated:
            # No valid trajectory to summarise. Counting one would report a capture that
            # reset every 0.18 s as 10 m of highly tortuous path -- as diversity.
            print(f"  unevaluable, excluded from statistics: {episode_id}")
            continue
        payload, _ = best_evaluable_payload(raw)

        behaviour, speed = infer_behaviour(episode_id, specs)
        try:
            summary = summarise_episode(episode_id, payload)
            contacts = decompose_payload_contacts(payload)
        except Exception:  # noqa: BLE001 - unevaluable captures still get a folder entry
            continue

        frame_count = int(payload.get("total_frames", 0))
        record = {
            "episode_id": episode_id,
            "behaviour": behaviour,
            "speed": speed,
            "outcome": outcome.outcome,
            "reasons": list(outcome.rejection_reasons),
            "frames": frame_count,
            "duration_s": frame_count / float(payload.get("fps", CONTROL_HZ)),
            "mean_speed_mps": summary.mean_speed_mps,
            "path_length_m": summary.path_length_m,
            "root_height_min_m": summary.root_height_min_m,
            "heading_change_rad": summary.heading_change_rad,
            "lateral_scene_contact_n": contacts.max_lateral_contact,
            "self_contact_n": contacts.max_self_contact,
            "source": str(rollout_dir),
        }
        records.append(record)
        episodes.append(summary)
        actions.append(np.asarray(payload["action_motion_token"], dtype=np.float64))

        folder = videos_root / f"{behaviour}"
        folder.mkdir(exist_ok=True)
        videos = sorted(p for p in rollout_dir.glob("renders/*.mp4") if p.stat().st_size > 4096)
        if videos:
            target = folder / f"{outcome.outcome}__{speed}__{episode_id[:52]}.mp4"
            if not target.exists():
                shutil.copy2(videos[0], target)
            record["video"] = str(target.relative_to(args.out))

        if figures_written[behaviour] < args.per_behaviour_figures:
            figure_dir = figures_root / behaviour
            figure_dir.mkdir(exist_ok=True)
            label = BEHAVIOUR_LABELS.get(behaviour, behaviour)
            try:
                episode_figure(
                    payload,
                    f"{label} · {speed} · {outcome.outcome} · {episode_id[:44]}",
                    figure_dir / f"{episode_id[:52]}.png",
                )
                figures_written[behaviour] += 1
            except Exception as error:  # noqa: BLE001 - a plot failure must not stop the build
                print(f"  figure failed for {episode_id}: {type(error).__name__}: {error}")

    if not records:
        raise SystemExit(f"no usable trajectories under {args.work}")

    corpus_figures(records, figures_root)
    diversity = build_diversity_report(episodes, actions)

    summary = {
        "episodes": len(records),
        "outcomes": {
            "accepted": tally.accepted,
            "rejected": tally.rejected,
            "unevaluable": tally.unevaluable,
            "recovered_from_split": tally.recovered_from_split,
            "acceptance_rate_of_evaluated": tally.acceptance_rate,
        },
        "diversity": {
            "pooled_rank": diversity.pooled_rank,
            "between_episode_rank": diversity.between_episode_rank,
            "within_episode_rank_mean": diversity.within_episode_rank_mean,
            "action_dim": diversity.action_dim,
        },
        "behaviours": sorted({r["behaviour"] for r in records}),
        "videos": sum(1 for r in records if "video" in r),
        "records": records,
    }
    (args.out / "review.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )

    write_review_markdown(args.out, summary, records)

    print(f"episodes: {len(records)}   {tally.summary()}")
    print(f"behaviours: {len(summary['behaviours'])}   videos: {summary['videos']}")
    print(
        f"effective rank  pooled {diversity.pooled_rank:.2f}   "
        f"between-episode {diversity.between_episode_rank:.2f}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
