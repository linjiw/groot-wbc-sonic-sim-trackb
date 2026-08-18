"""A structured prompt taxonomy for building a behaviour library, not a prompt list.

The corpus's measured weakness is behavioural: every accepted episode is a ~3.5 m forward
walk at ~0.7 m/s with a latent effective rank near 4. More episodes do not fix that --
more *kinds of motion* do. So prompts are generated from named axes rather than written
freehand, which buys three things a flat list cannot:

* **Coverage is auditable.** Each prompt carries the axis values that produced it, so
  "which speeds did we actually sample?" is a query, not a reading exercise.
* **Diversity is stratifiable.** Acceptance and behaviour metrics can be reported per axis
  value, which turns "the dataset is diverse" into a measurement per axis.
* **Failure is attributable.** When a family of prompts produces untrackable motion, the
  axis value is already attached, so the finding is "backward segments fail at speed",
  not "some prompts failed".

Prompt phrasing follows the model card's training description ("captured human body
motions with corresponding text descriptions"), so prompts are written as third-person
descriptions of a person -- the form the model was conditioned on -- rather than as
commands to a robot. The task language a VLA trains on is a separate string, bound at
capture time, and is deliberately not the same text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Iterable, Sequence

#: Locomotion speed. The bands are the **measured** interquartile range of achieved speed
#: over 114 generated motions, not a guess: an earlier version declared (0.2, 0.5) /
#: (0.5, 0.9) / (0.9, 1.4) and the generator ran faster than every one of those labels.
#:
#: The medians separate cleanly (0.68 / 0.93 / 1.26 m/s) and the interquartile bands do not
#: overlap, so the axis works. The full ranges overlap heavily -- "slow" reaches 1.25 m/s
#: and "brisk" descends to 0.77 -- so a speed style shifts the distribution rather than
#: fixing the speed. Treat these as where the mass lands, never as a guarantee about any
#: single sample.
SPEED_STYLES: dict[str, tuple[str, tuple[float, float]]] = {
    "slow": ("slowly", (0.59, 0.75)),
    "steady": ("at a steady pace", (0.83, 1.06)),
    "brisk": ("briskly", (1.09, 1.39)),
}

#: Heading change over the clip. "straight" is the corpus's entire current coverage.
TURN_STYLES: dict[str, str] = {
    "straight": "in a straight line",
    "gentle_left": "curving gently to the left",
    "gentle_right": "curving gently to the right",
    "sharp_left": ", then turns sharply left and continues",
    "sharp_right": ", then turns sharply right and continues",
}

#: What the body does besides translating. This is the axis that moves effective rank:
#: a crouch or a side-step uses joints a forward walk never leaves its narrow band on.
BODY_MODES: dict[str, str] = {
    "walk": "A person walks {speed} {turn}",
    "walk_pause": "A person walks {speed} {turn}, stops and stands still for a moment, then walks on",
    "walk_look": "A person walks {speed} {turn}, pauses and looks around, then continues walking",
    # "crouches down low" was screened out 3/3, limited by waist_pitch_joint: Kimodo folds
    # a human at the waist further than the G1's range allows, the joint clamps, and the
    # thigh ends up through the pelvis. Asking for a shallower crouch is the fix -- a
    # crouch the robot can hold is worth more to the corpus than one it cannot.
    "crouch_walk": "A person bends the knees and lowers slightly, then moves forward {speed} staying low",
    "crouch_deep": "A person crouches down low and moves forward {speed} while staying crouched",
    "duck_under": "A person walks {speed} {turn} and ducks down low to pass under an obstacle",
    "side_step": "A person steps sideways {speed} while facing forward",
    # The lateral geometry regime had no mode at all until these two. side_step looks like
    # it should serve it and does not: stepping sideways is translation, and a robot
    # side-stepping through a doorway is exactly as wide as one walking through it. Mining
    # the corpus made that concrete -- 60 of the refusals when pairing every side_step
    # against every plain walk were "min_half_width_m spread", the width simply never
    # changes. Passing a gap needs the silhouette to narrow, so these ask for that directly.
    "arm_tuck": "A person walks {speed} {turn} holding both arms tight against the body to "
                "fit through a narrow gap",
    "shoulder_turn": "A person walks {speed} {turn}, turning one shoulder forward and "
                     "twisting the torso sideways to slip through a narrow opening",
    "backward": "A person walks backwards {speed} {turn}",
    "stand_to_walk": "A person stands still, then begins walking {speed} {turn}",
    "walk_to_stop": "A person walks {speed} {turn} and comes to a stop",
    "step_over": "A person walks {speed} {turn} and lifts a leg high to step over something low",
    "turn_in_place": "A person stands and turns around in place, then walks {speed} forward",
    "carry_walk": "A person walks {speed} {turn} while carrying something in both hands",
    "reach_walk": "A person walks {speed} {turn}, then stops and reaches forward with one arm",
    "squat_pick": "A person walks {speed} forward, squats down to pick something up, then stands",
}

#: Body modes whose phrasing already fixes the heading, so pairing them with a turn style
#: produces contradictory text ("steps sideways, then turns sharply left and continues").
_TURN_INCOMPATIBLE = frozenset(
    {"side_step", "turn_in_place", "crouch_walk", "crouch_deep", "squat_pick"}
)

#: Modes whose text ends on a terminal action. A gentle curve still applies during the
#: walk that precedes it, but a sharp turn clause appended afterwards describes a third
#: action after the motion has already concluded, which reads as a run-on and is not the
#: behaviour the axis is trying to sample.
_SHARP_TURN_INCOMPATIBLE = frozenset({"walk_to_stop", "reach_walk"})

#: Modes that describe the body doing something other than plain locomotion. Tracked
#: separately because they are the ones that should move the latent's effective rank.
NON_WALK_MODES = frozenset(
    {
        "crouch_walk",
        "crouch_deep",
        "duck_under",
        "side_step",
        "arm_tuck",
        "shoulder_turn",
        "backward",
        "step_over",
        "turn_in_place",
        "carry_walk",
        "reach_walk",
        "squat_pick",
    }
)


@dataclass(frozen=True)
class PromptSpec:
    """One prompt plus the axis values that produced it."""

    prompt: str
    body_mode: str
    speed: str
    turn: str
    #: Expected speed band for the chosen speed style, in m/s. An *intent*, not a promise:
    #: whether the generator honours it is exactly what the sweep measures.
    expected_speed_mps: tuple[float, float]

    @property
    def is_non_walk(self) -> bool:
        return self.body_mode in NON_WALK_MODES

    @property
    def axis_key(self) -> str:
        return f"{self.body_mode}|{self.speed}|{self.turn}"


@dataclass
class Taxonomy:
    """A generated prompt set with its coverage counts."""

    specs: list[PromptSpec]
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def prompts(self) -> list[str]:
        return [spec.prompt for spec in self.specs]

    def by_axis(self, axis: str) -> dict[str, list[PromptSpec]]:
        grouped: dict[str, list[PromptSpec]] = {}
        for spec in self.specs:
            grouped.setdefault(getattr(spec, axis), []).append(spec)
        return grouped


def _compose(body_mode: str, speed: str, turn: str) -> str:
    template = BODY_MODES[body_mode]
    speed_phrase = SPEED_STYLES[speed][0]
    turn_phrase = TURN_STYLES[turn]

    if turn.startswith("sharp"):
        # These phrasings are clause continuations, so they attach after the template's
        # own text rather than filling the {turn} slot mid-sentence.
        text = template.format(speed=speed_phrase, turn="").rstrip().rstrip(",")
        text = f"{text}{turn_phrase}"
    else:
        text = template.format(speed=speed_phrase, turn=turn_phrase)

    text = " ".join(text.split())
    text = text.replace(" ,", ",")
    return text.rstrip(",. ")


def build_taxonomy(
    body_modes: Iterable[str] | None = None,
    speeds: Iterable[str] | None = None,
    turns: Iterable[str] | None = None,
) -> Taxonomy:
    """Cross the axes into prompts, skipping combinations that contradict themselves."""
    modes = list(body_modes) if body_modes is not None else list(BODY_MODES)
    speed_list = list(speeds) if speeds is not None else list(SPEED_STYLES)
    turn_list = list(turns) if turns is not None else list(TURN_STYLES)

    unknown = [m for m in modes if m not in BODY_MODES]
    if unknown:
        raise ValueError(f"unknown body mode(s): {unknown}")
    unknown = [s for s in speed_list if s not in SPEED_STYLES]
    if unknown:
        raise ValueError(f"unknown speed style(s): {unknown}")
    unknown = [t for t in turn_list if t not in TURN_STYLES]
    if unknown:
        raise ValueError(f"unknown turn style(s): {unknown}")

    specs: list[PromptSpec] = []
    seen: set[str] = set()
    for mode, speed, turn in itertools.product(modes, speed_list, turn_list):
        if mode in _TURN_INCOMPATIBLE and turn != "straight":
            continue
        if mode in _SHARP_TURN_INCOMPATIBLE and turn.startswith("sharp"):
            continue
        prompt = _compose(mode, speed, turn)
        if prompt in seen:
            continue
        seen.add(prompt)
        specs.append(
            PromptSpec(
                prompt=prompt,
                body_mode=mode,
                speed=speed,
                turn=turn,
                expected_speed_mps=SPEED_STYLES[speed][1],
            )
        )

    counts = {
        "body_mode": {k: len(v) for k, v in _group(specs, "body_mode").items()},
        "speed": {k: len(v) for k, v in _group(specs, "speed").items()},
        "turn": {k: len(v) for k, v in _group(specs, "turn").items()},
    }
    return Taxonomy(specs=specs, counts=counts)


def _group(specs: Sequence[PromptSpec], axis: str) -> dict[str, list[PromptSpec]]:
    grouped: dict[str, list[PromptSpec]] = {}
    for spec in specs:
        grouped.setdefault(getattr(spec, axis), []).append(spec)
    return grouped
