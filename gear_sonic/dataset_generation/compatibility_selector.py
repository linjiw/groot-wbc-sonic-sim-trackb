"""Choose the minimal adaptation a scene requires, by scoring scene-motion compatibility.

The task is deliberately *not* scene classification. A model that learns "room 7 wants the
crouch" would score well on any split that leaks scene identity and would teach nothing about
geometry. So the model here never sees a scene id: it sees a geometry profile and a candidate
motion's profile, and predicts whether that candidate survives that scene. Selection is then the
lexicographic rule -- among candidates predicted to succeed, take the cheapest -- which avoids
committing to adaptation-cost weights that are not yet frozen.

Because a compatibility model can only be trusted if it is *using* the scene, five controls ship
with the evaluator rather than as an afterthought. Two of them (no-scene, scene-shuffle) must
collapse to chance; if they do not, the reported accuracy is coming from somewhere else.

None of the metrics here constitute a result on their own. With a handful of families this is a
harness smoke test, and `SyntheticFamilies` exists so the controls can be shown to work before
real families are spent on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Order matters only for reproducibility of the feature vector.
SCENE_FEATURES = ("overhead_clearance_m", "left_gap_m", "right_gap_m", "floor_height_m")
MOTION_FEATURES = ("peak_height_m", "half_width_left_m", "half_width_right_m", "foot_clearance_m")


def _sigmoid(t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(t, -60.0, 60.0)))


@dataclass(frozen=True)
class Candidate:
    """One motion offered to the selector, with the cost of choosing it over the nominal."""

    motion_id: str
    #: Which nominal this derives from. Held out as a group so the selector cannot memorise
    #: "this walk always needs a crouch".
    nominal_id: str
    profile: dict[str, float]
    #: 0.0 for the nominal itself, by the lexicographic rule's definition.
    cost: float
    #: Whether physics accepted this candidate in this family's scene. Ground truth.
    succeeds: bool


@dataclass(frozen=True)
class Family:
    family_id: str
    scene: dict[str, float]
    candidates: tuple[Candidate, ...]

    def optimum(self) -> Candidate | None:
        """The cheapest candidate that actually succeeds, or None if the scene is impassable."""
        viable = [c for c in self.candidates if c.succeeds]
        return min(viable, key=lambda c: (c.cost, c.motion_id)) if viable else None


def features(
    scene: dict[str, float], motion: dict[str, float], *, blind: bool = False
) -> np.ndarray:
    """Feature vector for one (scene, candidate) pair.

    The margin terms are supplied explicitly. Handing the model `scene - motion` is a privileged
    representation and that is the point of the smoke stage: if a model *given* the margins
    cannot select correctly, the fault is in the harness, not in perception. `blind=True` zeroes
    every scene-derived term, which is the no-scene control.
    """
    m = np.asarray([motion[k] for k in MOTION_FEATURES], dtype=np.float64)
    if blind:
        return np.concatenate([m, np.zeros(len(SCENE_FEATURES) * 2 + 1)])
    s = np.asarray([scene[k] for k in SCENE_FEATURES], dtype=np.float64)
    margins = s - m
    # Sign flipped on the floor term: there the motion must exceed the obstacle, not clear it.
    signed = np.asarray([margins[0], margins[1], margins[2], -margins[3]])
    # The limiting margin, supplied explicitly. A candidate survives when *every* margin is
    # positive, and an AND of four conditions is not linearly separable -- a linear model given
    # only the four margins scored below the always-nominal baseline, choosing a failing
    # candidate 47% of the time. Handing over the minimum is what makes this model privileged,
    # which is the smoke stage's purpose: establish that the harness and the ceiling exist
    # before an ego-depth model is asked to recover the same quantity from pixels.
    return np.concatenate([m, s, signed, [signed.min()]])


class CompatibilityModel:
    """Logistic regression on margin features, fit by plain gradient descent.

    Deliberately the smallest thing that can express "succeeds when every margin is positive".
    A larger model would make it harder to tell whether a control collapsed for the right
    reason, and nothing here is a capacity experiment.
    """

    def __init__(self, *, seed: int = 0, steps: int = 4000, lr: float = 0.2) -> None:
        self.seed, self.steps, self.lr = seed, steps, lr
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CompatibilityModel":
        rng = np.random.default_rng(self.seed)
        self.mu, self.sigma = x.mean(0), x.std(0) + 1e-9
        z = (x - self.mu) / self.sigma
        self.w = rng.normal(0.0, 0.01, z.shape[1])
        for _ in range(self.steps):
            p = _sigmoid(z @ self.w + self.b)
            g = p - y
            self.w -= self.lr * (z.T @ g) / len(z)
            self.b -= self.lr * float(g.mean())
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mu) / self.sigma
        return _sigmoid(z @ self.w + self.b)


@dataclass
class SelectorMetrics:
    """What the selector got right, and the three distinct ways it can be wrong."""

    families: int = 0
    #: Chose exactly the cheapest succeeding candidate.
    choice_accuracy: float = 0.0
    #: Chose a candidate that physics rejects. The dangerous error.
    false_safe_rate: float = 0.0
    #: Paid for adaptation the scene did not require.
    unnecessary_adaptation_rate: float = 0.0
    #: Mean cost paid above the optimum, over families where the choice did succeed.
    cost_regret: float = 0.0
    #: Families the selector declared impassable. Counted separately from a wrong choice.
    abstention_rate: float = 0.0
    control: str = "none"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "control": self.control,
            "families": self.families,
            "choice_accuracy": round(self.choice_accuracy, 4),
            "false_safe_rate": round(self.false_safe_rate, 4),
            "unnecessary_adaptation_rate": round(self.unnecessary_adaptation_rate, 4),
            "cost_regret": round(self.cost_regret, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "notes": list(self.notes),
        }


def select(
    model: CompatibilityModel,
    family: Family,
    *,
    blind: bool = False,
    scene_override: dict[str, float] | None = None,
    order: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Candidate | None:
    """Apply the lexicographic rule using the model's success predictions.

    `order` permutes the candidate list before scoring. A selector whose answer depends on it is
    reading presentation rather than geometry, so the evaluator varies it as a control.
    """
    scene = scene_override if scene_override is not None else family.scene
    cands = list(family.candidates)
    if order is not None:
        cands = [cands[i] for i in order]
    x = np.stack([features(scene, c.profile, blind=blind) for c in cands])
    p = model.predict_proba(x)
    viable = [(c, q) for c, q in zip(cands, p) if q >= threshold]
    if not viable:
        return None
    return min(viable, key=lambda cq: (cq[0].cost, cq[0].motion_id))[0]


def evaluate(
    model: CompatibilityModel,
    families: list[Family],
    *,
    control: str = "none",
    seed: int = 0,
) -> SelectorMetrics:
    """Score the selector on held-out families under one control condition."""
    rng = np.random.default_rng(seed)
    scored = [f for f in families if f.optimum() is not None]
    notes: list[str] = []
    if len(scored) != len(families):
        notes.append(f"{len(families) - len(scored)} impassable families excluded from scoring")
    if not scored:
        return SelectorMetrics(control=control, notes=tuple(notes + ["no scorable families"]))

    shuffled = None
    if control == "scene-shuffle":
        # Pair each family with another family's scene. The candidates' ground truth still comes
        # from the real scene, so a selector that ignores geometry is unaffected -- which is
        # exactly what this control detects.
        perm = rng.permutation(len(scored))
        if len(scored) > 1:
            while np.any(perm == np.arange(len(scored))):
                perm = rng.permutation(len(scored))
        shuffled = [scored[i].scene for i in perm]

    correct = false_safe = unnecessary = abstained = 0
    regrets: list[float] = []
    for i, fam in enumerate(scored):
        order = rng.permutation(len(fam.candidates)) if control == "candidate-order" else None
        chosen = select(
            model,
            fam,
            blind=(control == "no-scene"),
            scene_override=shuffled[i] if shuffled else None,
            order=order,
        )
        best = fam.optimum()
        if chosen is None:
            abstained += 1
            continue
        if chosen.motion_id == best.motion_id:
            correct += 1
        if not chosen.succeeds:
            false_safe += 1
        else:
            regrets.append(chosen.cost - best.cost)
            if best.cost == 0.0 and chosen.cost > 0.0:
                unnecessary += 1

    n = len(scored)
    return SelectorMetrics(
        families=n,
        choice_accuracy=correct / n,
        false_safe_rate=false_safe / n,
        unnecessary_adaptation_rate=unnecessary / n,
        cost_regret=float(np.mean(regrets)) if regrets else 0.0,
        abstention_rate=abstained / n,
        control=control,
        notes=tuple(notes),
    )


def training_matrix(families: list[Family]) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for fam in families:
        for c in fam.candidates:
            x.append(features(fam.scene, c.profile))
            y.append(float(c.succeeds))
    return np.stack(x), np.asarray(y)


def holdout(
    families: list[Family], *, by: str, fold: int, folds: int = 4
) -> tuple[list[Family], list[Family]]:
    """Split families so a whole group is unseen at test time.

    `by="family"` holds out families; `by="nominal"` holds out every family whose candidates
    derive from a given nominal motion, which is the stricter test -- without it the selector can
    pass by memorising what each nominal usually needs.
    """
    if by == "family":
        keys = [f.family_id for f in families]
    elif by == "nominal":
        keys = [sorted({c.nominal_id for c in f.candidates})[0] for f in families]
    else:
        raise ValueError(f"unknown holdout axis {by!r}")
    groups = sorted(set(keys))
    test_groups = {g for i, g in enumerate(groups) if i % folds == fold}
    train = [f for f, k in zip(families, keys) if k not in test_groups]
    test = [f for f, k in zip(families, keys) if k in test_groups]
    return train, test
