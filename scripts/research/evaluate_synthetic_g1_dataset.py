#!/usr/bin/env python3
"""Semantic validity checks for a synthetic_g1 VLA dataset.

`check_sonic_vla_dataset.py` answers "is this well-formed?" -- shapes, dtypes,
frame counts, decodable video. That is necessary and not close to sufficient. A
dataset can pass every schema check and still be worthless, or actively harmful,
to train on. This script asks the questions that actually decide whether the data
teaches a policy anything true.

What "valid" means for an imitation dataset like this one
--------------------------------------------------------

1. **The images must not contain the answer.** Simulator debug overlays are the
   classic failure. A real instance was found in this pipeline: the motion
   command's ``debug_vis`` drew the reference-motion goal markers as bright
   yellow spheres, and the ego camera rendered them into the observation -- up to
   15% of the frame. A VLA trained on that can follow the markers instead of
   grounding language and scene, then fail completely at deployment, because the
   markers do not exist on a real robot. This is goal leakage, and no numeric
   schema check can see it.

2. **The camera must be attached to the robot.** If the view does not change when
   the body moves, the visual modality is decoration. Checked by correlating
   frame-to-frame image change against root speed.

3. **The action must carry information.** A near-constant action sequence teaches
   nothing regardless of how many rows it has. Checked by distinct-value count,
   per-dimension variance, and the participation ratio (effective rank) of the
   64-dim latent.

4. **The action must be coupled to what the robot does.** This is the strongest
   check here: fit a ridge regression from action to next-step state change, and
   compare against the same fit with actions shuffled across time. If shuffling
   does not hurt, the recorded action does not explain the recorded motion and
   the dataset cannot support behaviour cloning.

5. **Temporal alignment -- what this script CANNOT check.** Rows pair an observation
   with the action taken *after* it. Regression cannot verify that, and saying so is
   more useful than a check that gives false confidence. Negative controls on real
   data: reversing the one-frame shift moved R^2 by 0.015 (0.717 -> 0.724 the wrong
   way), and even a 25-frame (0.5 s) offset only moved it 0.717 -> 0.556, still far
   above the shuffled baseline of -0.04. The cause is that the 64-dim latent is
   smooth: autocorrelation r=0.981 at lag 1 and 0.874 at lag 5. Neighbouring actions
   are interchangeable to any regressor. The one-frame causal contract is therefore
   guaranteed structurally, by the exporter's shift and its sentinel tests, not here.
   The R^2-vs-lag numbers are still reported as diagnostics.

6. **Time must be continuous.** Uniform timestamps and no state discontinuity
   that implies impossible joint velocity.

7. **Duplicates vs visual variants.** Two episodes with the same state trajectory
   are one behaviour, however many rows they add. But if their imagery differs they
   are still useful as visual-robustness augmentation, which is exactly what a
   clutter-layout seed sweep produces. Measured on this dataset: two clutter seeds of
   one motion had bit-identical state (the furniture never touches the robot, so the
   physics is unchanged) and a mean per-pixel image difference of 21.2/255. Those are
   reported as visual variants and excluded from the distinct-behaviour count, not
   flagged as waste.

Usage::

    python scripts/research/evaluate_synthetic_g1_dataset.py /path/to/datasets/*/
    python scripts/research/evaluate_synthetic_g1_dataset.py DIR --json report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STATE_KEY = "observation.state"
ACTION_KEY = "action.motion_token"
REFERENCE_KEY = "reference.g1_qpos"
GRAVITY_KEY = "observation.projected_gravity"

#: Saturated primary colours used by Isaac Lab visualization markers. Scene content
#: in the M0 packages is desaturated (greys, muted blue, amber), so a large blob of
#: any of these is an overlay rather than geometry.
MARKER_COLORS = {
    "yellow": lambda r, g, b: (r > 150) & (g > 150) & (b < 100),
    "magenta": lambda r, g, b: (r > 150) & (b > 150) & (g < 100),
    "pure_green": lambda r, g, b: (g > 180) & (r < 90) & (b < 90),
    "pure_red": lambda r, g, b: (r > 200) & (g < 70) & (b < 70),
}
#: Fraction of a frame that a marker blob must cover before it is called
#: contamination. Scene trim contributes ~0.0002; the observed defect hit 0.15.
MARKER_FRAME_FRACTION = 0.01
#: Share of frames allowed to exceed that before the episode fails.
MARKER_EPISODE_FRACTION = 0.02


def _cell(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object and array.size == 1:
        array = np.asarray(array.reshape(-1)[0])
    return np.asarray(array, dtype=np.float64).reshape(-1)


def load_episode(dataset_dir: Path) -> dict:
    import pandas as pd

    parquet_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"no parquet under {dataset_dir / 'data'}")
    frames = [pd.read_parquet(path) for path in parquet_files]
    frame = frames[0] if len(frames) == 1 else frames[0].__class__(frames[0]).append(frames[1:])
    columns = {}
    for key in (STATE_KEY, ACTION_KEY, REFERENCE_KEY, GRAVITY_KEY):
        if key in frame.columns:
            columns[key] = np.stack([_cell(v) for v in frame[key]])
    timestamps = (
        np.asarray(frame["timestamp"], dtype=np.float64) if "timestamp" in frame.columns else None
    )
    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks_path = dataset_dir / "meta" / "tasks.jsonl"
    tasks = []
    if tasks_path.is_file():
        tasks = [
            json.loads(line)["task"] for line in tasks_path.read_text().splitlines() if line.strip()
        ]
    videos = sorted((dataset_dir / "videos").rglob("*.mp4"))
    return {
        "name": dataset_dir.name,
        "columns": columns,
        "timestamps": timestamps,
        "info": info,
        "tasks": tasks,
        "video": videos[0] if videos else None,
        "rows": len(frame),
    }


# --------------------------------------------------------------------------
# 1-2. Visual integrity and camera attachment
# --------------------------------------------------------------------------


def check_video(episode: dict, state: np.ndarray) -> dict:
    import cv2

    path = episode["video"]
    result: dict = {"checks": {}, "errors": [], "warnings": []}
    if path is None:
        result["errors"].append("episode has no video")
        return result

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        result["errors"].append("video decoded zero frames")
        return result
    video = np.stack(frames)
    red, green, blue = (video[..., i].astype(np.int16) for i in range(3))

    # 1. Marker / debug-overlay contamination.
    worst_name, worst_share, worst_frames = None, 0.0, 0
    for name, predicate in MARKER_COLORS.items():
        mask = predicate(red, green, blue)
        per_frame = mask.reshape(len(video), -1).mean(axis=1)
        share = float(per_frame.max())
        contaminated = int((per_frame > MARKER_FRAME_FRACTION).sum())
        if contaminated > worst_frames or (contaminated == worst_frames and share > worst_share):
            worst_name, worst_share, worst_frames = name, share, contaminated
    contaminated_fraction = worst_frames / len(video)
    result["checks"]["marker_contamination"] = {
        "worst_color": worst_name,
        "max_frame_share": worst_share,
        "frames_over_threshold": worst_frames,
        "fraction_of_frames": contaminated_fraction,
    }
    if contaminated_fraction > MARKER_EPISODE_FRACTION:
        result["errors"].append(
            f"simulator overlay contamination: {worst_frames}/{len(video)} frames have "
            f">{MARKER_FRAME_FRACTION:.0%} {worst_name} pixels (peak {worst_share:.1%}). "
            "Debug markers in the observation leak the tracking goal and do not exist "
            "at deployment; disable manager_env.commands.motion.debug_vis"
        )

    # 1b. Frozen or blank imagery.
    grey = video.mean(axis=-1)
    frame_delta = np.abs(np.diff(grey, axis=0)).mean(axis=(1, 2))
    frozen = int((frame_delta < 1e-6).sum())
    result["checks"]["frozen_frames"] = frozen
    result["checks"]["mean_frame_delta"] = float(frame_delta.mean())
    if frozen > len(video) * 0.1:
        result["errors"].append(f"{frozen}/{len(video)-1} consecutive frames are identical")
    if float(grey.std()) < 1.0:
        result["errors"].append("video is nearly constant; the camera may be blind")

    # 2. Camera attachment: image change must track how fast the body moves.
    if state is not None and len(state) >= len(frame_delta) + 1 and len(frame_delta) > 10:
        # Root motion is not in observation.state, so use whole-body configuration
        # change as the proprioceptive proxy for "how much the robot moved".
        motion = np.abs(np.diff(state[: len(frame_delta) + 1], axis=0)).mean(axis=1)
        if motion.std() > 1e-9 and frame_delta.std() > 1e-9:
            correlation = float(np.corrcoef(motion, frame_delta)[0, 1])
            result["checks"]["image_motion_correlation"] = correlation
            if correlation < 0.1:
                result["errors"].append(
                    f"ego imagery is uncorrelated with body motion (r={correlation:.3f}); "
                    "the camera may be detached from the robot"
                )
            elif correlation < 0.3:
                result["warnings"].append(
                    f"weak image/motion correlation (r={correlation:.3f})"
                )
    return result


# --------------------------------------------------------------------------
# 3-5. Action informativeness, coupling, and causal direction
# --------------------------------------------------------------------------


def _ridge_r2(inputs: np.ndarray, targets: np.ndarray, penalty: float = 1.0) -> float:
    """Held-out R^2 of a ridge map from inputs to targets.

    The split is INTERLEAVED (even rows fit, odd rows score), not temporal. A
    walking trajectory is strongly non-stationary, so a first-half/second-half
    split measures whether a linear map extrapolates across the episode -- which
    is not the question, and which produced R^2 of -1.1 on data that is in fact
    well coupled. Interleaving keeps both folds on the same distribution so the
    comparison against the shuffled control is meaningful. Adjacent frames are
    correlated, so this is a coupling test, not a generalisation estimate.
    """
    if len(inputs) < 20:
        return float("nan")
    train_x, train_y = inputs[0::2], targets[0::2]
    test_x, test_y = inputs[1::2], targets[1::2]
    x_mean, y_mean = train_x.mean(axis=0), train_y.mean(axis=0)
    centered_x, centered_y = train_x - x_mean, train_y - y_mean
    gram = centered_x.T @ centered_x + penalty * np.eye(centered_x.shape[1])
    weights = np.linalg.solve(gram, centered_x.T @ centered_y)
    prediction = (test_x - x_mean) @ weights + y_mean
    residual = float(((test_y - prediction) ** 2).sum())
    total = float(((test_y - test_y.mean(axis=0)) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def check_action(action: np.ndarray, state: np.ndarray) -> dict:
    result: dict = {"checks": {}, "errors": [], "warnings": []}

    # 3. Informativeness.
    distinct = int(np.unique(action).size)
    per_dim_std = action.std(axis=0)
    dead_dims = int((per_dim_std < 1e-9).sum())
    eigenvalues = np.linalg.eigvalsh(np.cov(action.T) + 1e-12 * np.eye(action.shape[1]))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    participation = (
        float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum()) if eigenvalues.sum() > 0 else 0.0
    )
    result["checks"]["action_distinct_values"] = distinct
    result["checks"]["action_dead_dims"] = dead_dims
    result["checks"]["action_effective_rank"] = participation
    result["checks"]["action_mean_std"] = float(per_dim_std.mean())
    if distinct < 3:
        result["errors"].append(
            f"action takes only {distinct} distinct values; it carries no information"
        )
    if participation < 2.0:
        result["errors"].append(
            f"action effective rank {participation:.2f}: the 64-dim latent is nearly one-dimensional"
        )
    if dead_dims > action.shape[1] // 2:
        result["warnings"].append(f"{dead_dims}/{action.shape[1]} action dims are constant")

    # 4-5. Coupling to dynamics, and causal direction.
    if state is not None and len(state) == len(action) and len(state) > 40:
        future = np.diff(state, axis=0)  # state[t+1] - state[t]
        actions_now = action[:-1]
        forward = _ridge_r2(actions_now, future)
        rng = np.random.default_rng(0)
        shuffled = _ridge_r2(actions_now[rng.permutation(len(actions_now))], future)
        backward = _ridge_r2(action[1:], future)  # action taken AFTER the transition
        result["checks"]["action_to_next_state_r2"] = forward
        result["checks"]["shuffled_action_r2"] = shuffled
        result["checks"]["action_to_previous_state_r2"] = backward

        # Reported as diagnostics only. Regression on this latent CANNOT detect a
        # temporal offset; see the module docstring for the measurements.
        lag = min(25, len(actions_now) // 4)
        far = _ridge_r2(actions_now[lag:], future[: len(future) - lag]) if lag >= 5 else float("nan")
        result["checks"]["decorrelation_lag_frames"] = lag
        result["checks"]["far_future_action_r2"] = far

        if not math.isnan(forward):
            if forward <= max(shuffled, 0.0) + 0.02:
                result["errors"].append(
                    f"action does not explain the motion: R^2={forward:.3f} vs shuffled "
                    f"{shuffled:.3f}. Behaviour cloning cannot learn from this pairing"
                )
            # NOTE: deliberately not an error. Negative controls showed this family of
            # metrics cannot detect temporal offset at all -- see the module docstring.
    return result


# --------------------------------------------------------------------------
# 6. Temporal continuity
# --------------------------------------------------------------------------


def check_continuity(episode: dict, state: np.ndarray, fps: float) -> dict:
    result: dict = {"checks": {}, "errors": [], "warnings": []}
    timestamps = episode["timestamps"]
    if timestamps is not None and len(timestamps) > 2:
        deltas = np.diff(timestamps)
        jitter = float(np.max(np.abs(deltas - 1.0 / fps)))
        result["checks"]["timestamp_max_jitter_s"] = jitter
        if np.any(deltas <= 0):
            result["errors"].append("timestamps are not strictly increasing")
        if jitter > 1e-3:
            result["errors"].append(f"timestamp jitter {jitter:.4f}s exceeds 1 ms")
    if state is not None and len(state) > 2:
        # Implied joint speed between consecutive rows. G1 joints do not exceed a
        # few tens of rad/s; anything far beyond that is a discontinuity, not motion.
        speed = np.abs(np.diff(state, axis=0)) * fps
        peak = float(speed.max())
        result["checks"]["max_implied_joint_speed_rad_s"] = peak
        if peak > 100.0:
            result["errors"].append(
                f"state discontinuity: implied joint speed {peak:.1f} rad/s between rows"
            )
        elif peak > 50.0:
            result["warnings"].append(f"high implied joint speed {peak:.1f} rad/s")
    return result


# --------------------------------------------------------------------------
# 7. Cross-episode duplication and diversity
# --------------------------------------------------------------------------


def _video_difference(first: Path | None, second: Path | None, frames: int = 40) -> float | None:
    """Mean per-pixel difference between the first `frames` of two videos."""
    if first is None or second is None:
        return None
    import cv2

    stacks = []
    for path in (first, second):
        capture = cv2.VideoCapture(str(path))
        collected = []
        try:
            for _ in range(frames):
                ok, image = capture.read()
                if not ok:
                    break
                collected.append(image)
        finally:
            capture.release()
        if not collected:
            return None
        stacks.append(np.stack(collected).astype(np.int16))
    count = min(len(stacks[0]), len(stacks[1]))
    return float(np.abs(stacks[0][:count] - stacks[1][:count]).mean())


def check_diversity(episodes: list[dict]) -> dict:
    result: dict = {"checks": {}, "errors": [], "warnings": []}
    _episode_by_name = {episode["name"]: episode for episode in episodes}
    signatures = {}
    for episode in episodes:
        state = episode["columns"].get(STATE_KEY)
        if state is None or not len(state):
            continue
        length = min(200, len(state))
        signatures[episode["name"]] = state[:length].astype(np.float64).ravel()

    names = list(signatures)
    duplicates = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = signatures[first], signatures[second]
            size = min(len(a), len(b))
            if size == 0:
                continue
            difference = float(np.abs(a[:size] - b[:size]).mean())
            if difference < 1e-4:
                duplicates.append((first, second, difference))
    # A pair with identical state but different imagery is NOT a useless duplicate: it
    # is the same behaviour in a different visual context, which is exactly what a
    # clutter-layout seed sweep produces and is useful for visual robustness. It must
    # still not be counted as behavioural diversity, so it is reported separately.
    visual_variants, true_duplicates = [], []
    for first, second, difference in duplicates:
        image_delta = _video_difference(
            _episode_by_name[first].get("video"), _episode_by_name[second].get("video")
        )
        record = [first, second, difference, image_delta]
        (visual_variants if (image_delta or 0.0) > 2.0 else true_duplicates).append(record)

    result["checks"]["episode_count"] = len(episodes)
    result["checks"]["identical_state_pairs"] = [
        [first, second, difference] for first, second, difference, _ in visual_variants + true_duplicates
    ]
    result["checks"]["visual_variant_pairs"] = visual_variants
    result["checks"]["true_duplicate_pairs"] = true_duplicates
    result["checks"]["distinct_behaviours"] = len(episodes) - len(duplicates)
    if true_duplicates:
        result["errors"].append(
            f"{len(true_duplicates)} fully duplicated episode pair(s) (identical state AND "
            "imagery); this inflates apparent dataset size without adding information"
        )
    if visual_variants:
        result["warnings"].append(
            f"{len(visual_variants)} pair(s) share an identical state/action trajectory but "
            "differ visually. Useful for visual robustness, but they are ONE behaviour: "
            f"{len(episodes)} episodes cover {len(episodes) - len(duplicates)} distinct behaviours"
        )

    tasks = sorted({task for episode in episodes for task in episode["tasks"]})
    result["checks"]["distinct_tasks"] = tasks
    if not tasks:
        result["errors"].append("no language annotation found in any episode")
    elif len(tasks) == 1 and len(episodes) > 1:
        result["warnings"].append(
            f"all {len(episodes)} episodes share one task string {tasks[0]!r}; a VLA cannot "
            "learn language grounding from a single instruction"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+", help="exported dataset directories")
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--skip-video", action="store_true", help="skip decode-heavy checks")
    args = parser.parse_args()

    directories = []
    for entry in args.datasets:
        if (entry / "meta" / "info.json").is_file():
            directories.append(entry)
        else:
            directories.extend(
                sorted(child for child in entry.iterdir() if (child / "meta").is_dir())
            )
    if not directories:
        print("ERROR: no exported datasets found", file=sys.stderr)
        return 2

    episodes, reports = [], {}
    total_errors = total_warnings = 0
    for directory in directories:
        episode = load_episode(directory)
        episodes.append(episode)
        state = episode["columns"].get(STATE_KEY)
        action = episode["columns"].get(ACTION_KEY)
        fps = float(episode["info"].get("fps", 50.0))

        report = {"rows": episode["rows"], "checks": {}, "errors": [], "warnings": []}
        for part in (
            check_action(action, state) if action is not None else {"errors": ["missing action"]},
            check_continuity(episode, state, fps),
            ({"checks": {}, "errors": [], "warnings": []} if args.skip_video
             else check_video(episode, state)),
        ):
            report["checks"].update(part.get("checks", {}))
            report["errors"].extend(part.get("errors", []))
            report["warnings"].extend(part.get("warnings", []))
        reports[episode["name"]] = report
        total_errors += len(report["errors"])
        total_warnings += len(report["warnings"])

        status = "FAIL" if report["errors"] else ("warn" if report["warnings"] else "ok")
        print(f"[{status:>4}] {episode['name']}  ({report['rows']} rows)")
        for error in report["errors"]:
            print(f"         ERROR: {error}")
        for warning in report["warnings"]:
            print(f"         warn:  {warning}")

    diversity = check_diversity(episodes)
    total_errors += len(diversity["errors"])
    total_warnings += len(diversity["warnings"])
    print(f"\n[dataset-level] {len(episodes)} episodes")
    for error in diversity["errors"]:
        print(f"         ERROR: {error}")
    for warning in diversity["warnings"]:
        print(f"         warn:  {warning}")
    print(f"         distinct tasks: {diversity['checks']['distinct_tasks']}")

    print(f"\n{'PASS' if total_errors == 0 else 'FAIL'}: "
          f"{total_errors} error(s), {total_warnings} warning(s)")
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(
                {"episodes": reports, "dataset": diversity, "errors": total_errors},
                indent=2,
                sort_keys=True,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_path}")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
