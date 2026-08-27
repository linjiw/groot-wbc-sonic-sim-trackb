"""Multi-obstacle LfLH for humanoid whole-body counterfactuals.

The first attempt at this (``learned_hallucinator.py``, retracted) reduced the problem to two
parameters and a decoder that was a soft indicator of a closed-form interval. A review showed the
result was an artifact: the model was a constant function of its input, its variance sat on a
clamp, and the decoder encoded the answer it was supposed to recover.

This is the real thing, built to the structure LfLH actually uses:

    trajectory  ->  hallucinator  ->  K obstacles  ->  fixed differentiable decoder  ->  choice
                                                                                          |
                                    reconstruction loss <---------------------------------+

with three properties the retracted version lacked.

**The decoder decides something.** It does not evaluate a known interval. Given obstacles it scores
*every* candidate motion — the nominal, crouches at several depths, one-sided arm tucks — for
feasibility, and returns a soft-argmin over edit cost among the survivors. Reconstruction asks the
observed motion to come out as the winner. Obstacles can only change that by actually blocking some
candidates and not others, which is the property that stops encoder and decoder colluding.

**The inverse set is genuinely multimodal.** An overhead bar explains a crouch; a left-side
obstacle explains a left arm tuck; a right-side obstacle explains a right one; and several
obstacle sets explain the same observation. There is no closed form to leak, so coverage of the
inverse set is a real question rather than a tautology.

**The losses are LfLH's**, including the two the retracted version omitted: a genuine Gaussian KL
on obstacle size (which carries ``-log sigma`` and therefore opposes contraction) and an
obstacle-obstacle repulsion. Without them collapse is a theorem, not a finding.

Nothing here is a physics verdict. The decoder is a differentiable model of the *choice* rule; the
frozen controller in Isaac remains the only thing that decides whether a motion actually executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

#: Obstacle parameters, per obstacle, in the route frame of the binding station grid:
#: station index, lateral offset (signed, +left), vertical centre, and the three half extents.
OBSTACLE_PARAMS = 6


@dataclass(frozen=True)
class SceneSampleBatch:
    """Sampled obstacle sets for one trajectory: (samples, obstacles, OBSTACLE_PARAMS)."""

    motion_id: str
    parameters: np.ndarray
    winner: np.ndarray
    observed_index: int

    @property
    def reconstruction_rate(self) -> float:
        """Share of sampled scenes under which the observed motion is the preferred choice."""
        return float((self.winner == self.observed_index).mean())


class MultiObstacleHallucinator(nn.Module):
    """Conv1D over the directional extent profile -> Gaussians over K obstacles.

    Mirrors LfLH's encoder shape (three temporal convolutions, then a fully connected head emitting
    means and log-variances) with the obstacle parameterisation adapted from 2-D ellipses to
    route-frame boxes, since the humanoid case needs a height and two independent lateral sides.
    """

    def __init__(
        self,
        stations: int,
        obstacles: int = 6,
        hidden: int = 64,
        min_log_sigma: float = -20.0,
    ):
        super().__init__()
        self.stations = stations
        self.obstacles = obstacles
        # Deliberately permissive: a floor near zero must never be what a reported variance means.
        # The retracted result's headline was exactly its clamp.
        self.min_log_sigma = min_log_sigma
        # One shared encoder applied to the nominal and to the adapted motion, then fused on
        # their difference. Conditioning on the *pair* rather than on the observed motion alone
        # is the difference between asking "what scene explains this body" -- which requires the
        # model to first infer which edit it is looking at -- and "what scene explains this
        # change", where the edit is handed over explicitly. Measured: the single-motion encoder
        # was beaten by its own input-ablation, i.e. it never extracted the edit.
        self.encoder = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # [Z0, Z1, Z1-Z0, |Z1-Z0|] over both mean- and max-pooled features.
        self.head = nn.Sequential(
            nn.Linear(8 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * obstacles * OBSTACLE_PARAMS),
        )

    def _embed(self, profile: torch.Tensor) -> torch.Tensor:
        features = self.encoder(profile)
        return torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=-1)

    def seed(self, latent: np.ndarray, log_sigma: float = -1.0) -> None:
        """Start the output head at a known-good placement instead of at random.

        The final layer's weights are zeroed and its bias set to ``latent``, so the model's initial
        output *is* the seed for every input and the encoder begins by learning deviations from it.
        Without this the optimiser has to find a band tens of millimetres wide inside a range of
        hundreds, from a random start, across a loss that is flat outside the band.
        """
        final = self.head[-1]
        with torch.no_grad():
            final.weight.zero_()
            bias = torch.cat(
                (
                    torch.tensor(latent, dtype=torch.float32).reshape(-1),
                    torch.full((self.obstacles * OBSTACLE_PARAMS,), float(log_sigma)),
                )
            )
            final.bias.copy_(bias)

    def forward(self, profiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``profiles`` is (batch, 2, 3, stations): the nominal and the observed motion."""
        if profiles.dim() != 4 or profiles.shape[1] != 2:
            raise ValueError("pair-conditioned input must be (batch, 2, 3, stations)")
        nominal = self._embed(profiles[:, 0])
        adapted = self._embed(profiles[:, 1])
        pooled = torch.cat((nominal, adapted, adapted - nominal, (adapted - nominal).abs()), dim=-1)
        raw = self.head(pooled)
        batch = raw.shape[0]
        raw = raw.view(batch, self.obstacles, 2 * OBSTACLE_PARAMS)
        mean = raw[..., :OBSTACLE_PARAMS]
        log_sigma = raw[..., OBSTACLE_PARAMS:].clamp(min=self.min_log_sigma, max=2.0)
        return mean, log_sigma


@dataclass(frozen=True)
class PlausibleShape:
    """What a binding face should look like: thin along route and vertically, wide across it.

    These are the dimensions of the objects the corpus actually authors -- planks, lintels, beams,
    ducts -- not a constraint the inverse problem needs. They are applied as an annealed penalty so
    that plausibility is bought after the mechanism works, not imposed before it does.
    """

    half_along_range_m: tuple[float, float] = (0.03, 0.35)
    half_lateral_range_m: tuple[float, float] = (0.15, 1.20)
    half_vertical_range_m: tuple[float, float] = (0.02, 0.30)


@dataclass(frozen=True)
class ObstacleGeometry:
    """Latent -> metres. Kept separate so the parameterisation can be audited on its own.

    None of these anchors is derived from a feasibility answer. The retracted version anchored its
    coordinate on ``adapted.mean()``, which *is* the window's lower edge, so a perfect validity
    score was the parameterisation rather than the model.
    """

    stations: int
    height_centre_m: float = 1.30
    height_scale_m: float = 0.45
    lateral_scale_m: float = 0.60
    # Per-axis extent ranges, because the three axes of a binding face are not interchangeable.
    # A shared 0.04-0.80 m range let the model emit 1.6 m cubes that satisfied the decoder and
    # looked nothing like anything a person would duck under: the face must be *thin* along the
    # route and vertically, and may be *wide* across it.
    min_half_extent_m: float = 0.02
    max_half_extent_m: float = 1.20
    # Permissive superset: the search space in which the inverse problem is tractable. Plausible
    # shape is imposed by `PlausibleShape` through an annealed penalty, not by narrowing this.
    half_along_range_m: tuple[float, float] = (0.03, 0.80)
    half_lateral_range_m: tuple[float, float] = (0.04, 1.20)
    half_vertical_range_m: tuple[float, float] = (0.02, 0.80)

    def size_penalty(
        self, boxes: dict[str, torch.Tensor], target: "PlausibleShape"
    ) -> torch.Tensor:
        """How far outside a plausible shape these boxes are, in squared metres.

        Kept as a *penalty* rather than a narrowing of the parameterisation. Annealing the ranges
        themselves was tried and fails: the latent-to-metres map is a sigmoid onto the range, so
        moving the range remaps every learned latent mid-training and the solution is lost --
        measured, the binding face collapsed to 0.06 x 0.08 x 0.04 m. A penalty leaves the map
        stationary and lets the prior tighten around a solution the model already has.
        """
        cost = torch.zeros((), dtype=torch.float32)
        for key, (low, high) in (
            ("half_along_m", target.half_along_range_m),
            ("half_lateral_m", target.half_lateral_range_m),
            ("half_vertical_m", target.half_vertical_range_m),
        ):
            value = boxes[key]
            cost = cost + (torch.relu(value - high) ** 2 + torch.relu(low - value) ** 2).mean()
        return cost

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        station = torch.sigmoid(latent[..., 0]) * (self.stations - 1)
        lateral = torch.tanh(latent[..., 1]) * self.lateral_scale_m
        height = self.height_centre_m + torch.tanh(latent[..., 2]) * self.height_scale_m
        gates = torch.sigmoid(latent[..., 3:6])
        ranges = (
            self.half_along_range_m,
            self.half_lateral_range_m,
            self.half_vertical_range_m,
        )
        extents = [low + (high - low) * gates[..., axis] for axis, (low, high) in enumerate(ranges)]
        return {
            "station": station,
            "lateral_m": lateral,
            "height_m": height,
            "half_along_m": extents[0],
            "half_lateral_m": extents[1],
            "half_vertical_m": extents[2],
        }


def closed_form_seed(
    extents: np.ndarray,
    costs: np.ndarray,
    observed_index: int,
    geometry: "ObstacleGeometry",
    *,
    obstacles: int,
    clearance_m: float = 0.015,
) -> np.ndarray:
    """Where an obstacle *should* go, computed rather than searched.

    For the observed candidate, find the (direction, station) at which it has the largest margin
    over the cheapest competitor that would otherwise be chosen, and place a face in the middle of
    that margin. This is the same quantity the overhead window solver computes, generalised to
    left and right, and it exists precisely because the inverse problem is solvable pointwise even
    where it is hard to search.

    Returns latents, not metres, so the result can seed the network's output head directly.

    Seeding is legitimate only if it is reported: the model then *refines* a known-good placement
    rather than discovering it, and any claim about what the model learned must be made against a
    control that keeps the seed and drops the learning.
    """
    candidates, directions, stations = extents.shape
    # Competitors are every candidate at most as expensive as the observed one: those are the
    # motions the scene has to rule out, since a cheaper feasible option would win instead.
    competitors = [
        index
        for index in range(candidates)
        if index != observed_index and costs[index] <= costs[observed_index] + 1e-9
    ]
    if not competitors:
        competitors = [index for index in range(candidates) if index != observed_index]

    observed = extents[observed_index]
    rival = extents[competitors].min(axis=0)
    # Margin: how much *less* far the observed motion reaches than every cheaper rival.
    margin = rival - observed

    best = np.unravel_index(int(np.argmax(margin)), margin.shape)
    direction, station = int(best[0]), int(best[1])
    width = float(margin[direction, station])
    if width <= 2 * clearance_m:
        direction, station, width = 0, stations // 2, max(width, 4 * clearance_m)

    # Place the face just above the observed motion rather than at the middle of the band. The
    # midpoint splits the margin evenly, which leaves a cheaper rival penetrating by only half the
    # band -- often too little for the decision to overcome its cost advantage. Hugging the
    # observed motion makes every rival penetrate by nearly the whole band, which is the placement
    # that maximises the decision margin. It is also the minimum-regret placement, since regret is
    # exactly the observed motion's clearance under the face.
    face = float(observed[direction, station]) + max(clearance_m, 0.15 * width)

    def _logit(value: float, low: float = 1e-4) -> float:
        value = float(np.clip(value, low, 1.0 - low))
        return float(np.log(value / (1.0 - value)))

    def _atanh(value: float) -> float:
        return float(np.arctanh(np.clip(value, -0.999, 0.999)))

    latent = np.zeros((obstacles, OBSTACLE_PARAMS), dtype=np.float64)
    latent[:, 0] = _logit(station / max(stations - 1, 1))

    def _gate(value: float, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return _logit((float(np.clip(value, low, high)) - low) / max(high - low, 1e-9))

    latent[:, 3] = _gate(0.12, geometry.half_along_range_m)
    latent[:, 5] = _gate(0.08, geometry.half_vertical_range_m)

    if direction == 0:  # overhead: a face the observed motion passes under
        latent[:, 1] = 0.0
        latent[:, 2] = _atanh((face + 0.08 - geometry.height_centre_m) / geometry.height_scale_m)
        latent[:, 4] = _gate(0.90, geometry.half_lateral_range_m)
    else:  # lateral: a face the observed motion passes beside, on the side it narrowed
        sign = 1.0 if direction == 1 else -1.0
        half_lateral = 0.10
        latent[:, 1] = _atanh(sign * (face + half_lateral) / geometry.lateral_scale_m)
        latent[:, 2] = _atanh((0.95 - geometry.height_centre_m) / geometry.height_scale_m)
        latent[:, 4] = _gate(half_lateral, geometry.half_lateral_range_m)
        latent[:, 5] = _gate(0.30, geometry.half_vertical_range_m)
    # Only the first obstacle is seeded onto the binding face. The rest must start *harmless*, and
    # "neutral" latents are not: zeros decode to 1.7 m boxes centred on the route, which block
    # every candidate and make the deepest edit win regardless of the target. They are instead
    # placed at the lateral limit with minimum extent, where they cover neither the centreline nor
    # the body, and the model is free to bring them in.
    if obstacles > 1:
        extras = obstacles - 1
        latent[1:, 0] = np.linspace(-1.5, 1.5, extras)
        latent[1:, 1] = np.where(np.arange(extras) % 2 == 0, 3.0, -3.0)
        latent[1:, 2] = 0.0
        latent[1:, 3:6] = _logit(1e-3)  # minimum extent on every axis
    return latent


def relax(boxes: dict[str, torch.Tensor], *, margin_m: float = 0.10) -> dict[str, torch.Tensor]:
    """The easy scene: the same obstacles, moved clear of every candidate.

    A counterfactual is a *pair* of scenes. Training only the hard one rewards putting something
    large near the body; training the easy one alongside it requires the constraint to be
    releasable, which is what makes the hard scene's obstacle the thing that mattered rather than
    merely something that was present.
    """
    eased = dict(boxes)
    eased["height_m"] = boxes["height_m"] + margin_m
    eased["lateral_m"] = boxes["lateral_m"] + torch.sign(boxes["lateral_m"]) * margin_m
    return eased


@dataclass
class ChoiceDecoder:
    """Fixed, parameter-free, differentiable model of the *choice* the scene forces.

    For each candidate motion it computes a soft blocked-ness: how deeply the candidate's body
    would have to intrude into each obstacle. Blocking makes a candidate expensive; among those
    left, the cheapest edit wins. Returns a soft distribution over candidates.

    It has no learnable parameters, and — unlike the retracted surrogate — no knowledge of which
    candidate is the observed one.
    """

    temperature_m: float = 0.02
    station_sigma: float = 0.8
    choice_temperature: float = 0.15
    blocked_penalty: float = 60.0

    def penetration(
        self, extents: torch.Tensor, obstacles: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Soft intrusion of each candidate into each obstacle.

        ``extents`` is (candidates, 3, stations) holding up/left/right; the obstacle tensors are
        (obstacles,). Returns (candidates,): how much this scene blocks each candidate.

        An obstacle binds a candidate in exactly one of three ways, and which one depends on where
        the obstacle is, not on which candidate is being scored:

        * **overhead** -- the obstacle hangs above leg height and the body reaches above its
          underside;
        * **left** / **right** -- the obstacle straddles mid-body height on one side, and the body
          sticks out past its near face on that side.

        That asymmetry is what makes a left-side obstacle explain a *left* arm tuck and nothing
        else, and it is why the scene can respond to which way the motion leans.
        """
        candidates, _, stations = extents.shape
        index = torch.arange(stations, device=extents.device, dtype=extents.dtype)

        # How much of each obstacle sits at each station: (obstacles, stations).
        station_gap = (index[None, :] - obstacles["station"][:, None]).abs()
        span = obstacles["half_along_m"][:, None] / max(self.station_sigma, 1e-6)
        weight = torch.sigmoid((span - station_gap) / 0.35)

        up = extents[:, 0, :]
        left = extents[:, 1, :]
        right = extents[:, 2, :]

        low = obstacles["height_m"] - obstacles["half_vertical_m"]
        high = obstacles["height_m"] + obstacles["half_vertical_m"]

        # Whether the box spans the route centreline. This is the condition that separates
        # "something you duck under" from "something you pass beside": a box at lateral 0.38 with
        # half-width 0.10 covers [0.28, 0.48] and never sits above the head, so scoring it as
        # overhead made every side obstacle block every candidate and the decoder always chose the
        # nominal.
        covers_centre = torch.sigmoid(
            (obstacles["half_lateral_m"] - obstacles["lateral_m"].abs()) / 0.05
        )

        # Overhead depth, gated on being aloft *and* over the centreline.
        aloft = torch.sigmoid((low - 0.70) / 0.10)
        overhead = (up[:, None, :] - low[None, :, None]) * (aloft * covers_centre)[None, :, None]

        # Lateral depth against whichever side the obstacle occupies.
        left_face = obstacles["lateral_m"] - obstacles["half_lateral_m"]
        right_face = -(obstacles["lateral_m"] + obstacles["half_lateral_m"])
        depth_left = left[:, None, :] - left_face[None, :, None]
        depth_right = right[:, None, :] - right_face[None, :, None]
        on_left = torch.sigmoid(obstacles["lateral_m"] / 0.05)
        lateral = on_left[None, :, None] * depth_left + (1.0 - on_left)[None, :, None] * depth_right
        # It binds laterally only if it spans the body's height and sits off to one side.
        straddles = torch.sigmoid((high - 0.35) / 0.15) * torch.sigmoid((1.45 - low) / 0.15)
        lateral = lateral * (straddles * (1.0 - covers_centre))[None, :, None]

        depth = torch.maximum(overhead, lateral)
        soft = torch.nn.functional.softplus(depth / self.temperature_m) * self.temperature_m
        # Worst station per obstacle, summed over obstacles: several obstacles block more.
        return (soft * weight[None, :, :]).amax(dim=-1).sum(dim=-1).view(candidates)

    def __call__(
        self, extents: torch.Tensor, obstacles: dict[str, torch.Tensor], costs: torch.Tensor
    ) -> torch.Tensor:
        blocked = self.penetration(extents, obstacles)
        objective = costs + self.blocked_penalty * blocked
        return torch.softmax(-objective / self.choice_temperature, dim=0)


@dataclass
class TrainingReport:
    steps: int
    final_loss: float
    reconstruction: float
    mean_sigma: float
    history: list[dict] = field(default_factory=list)


def _kl(mean: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Genuine Gaussian KL to a unit prior; carries ``-log sigma`` and so opposes collapse."""
    return (0.5 * (mean.pow(2) + (2.0 * log_sigma).exp() - 1.0) - log_sigma).mean()


def train(
    batch_extents: list[np.ndarray],
    batch_costs: list[np.ndarray],
    observed: list[int],
    *,
    obstacles: int = 6,
    steps: int = 400,
    samples: int = 6,
    learning_rate: float = 2e-3,
    kl_weight: float = 0.02,
    easy_weight: float = 0.3,
    minimality_weight: float = 0.02,
    clearance_weight: float = 1.0,
    repulsion_weight: float = 0.2,
    clearance_m: float = 0.05,
    seed: int = 0,
    geometry: ObstacleGeometry | None = None,
    decoder: ChoiceDecoder | None = None,
    anneal_from_m: float | None = 0.15,
    anneal_prior: bool = False,
    shape_weight: float = 6.0,
    shape: "PlausibleShape | None" = None,
    seed_latents: list[np.ndarray] | None = None,
) -> tuple[MultiObstacleHallucinator, TrainingReport]:
    """LfLH's objective: reconstruct the observed choice, with size KL, clearance and repulsion.

    ``anneal_from_m`` starts the decoder soft and sharpens it to its configured temperature over
    training. A sharp decoder is the faithful one, but its gradient is flat wherever a candidate is
    comfortably blocked or comfortably clear, which is everywhere outside a band a few centimetres
    wide. Annealing gives the optimiser a signal to follow in before the decision is made crisp.
    LfLH anneals its own loss weights over 1000 epochs for the same reason.

    ``seed_latents`` starts the output head at a computed placement; see `closed_form_seed`.
    """
    torch.manual_seed(seed)
    stations = batch_extents[0].shape[-1]
    geometry = geometry or ObstacleGeometry(stations=stations)
    decoder = decoder or ChoiceDecoder()
    extents = [torch.tensor(item, dtype=torch.float32) for item in batch_extents]
    costs = [torch.tensor(item, dtype=torch.float32) for item in batch_costs]
    # The hallucinator sees only the observed motion, as LfLH sees only the executed plan.
    # Pair conditioning: the nominal and the observed motion, so the edit is explicit rather
    # than something the encoder has to infer from absolute extents.
    profiles = torch.stack(
        [
            torch.stack((extents[row][0], extents[row][observed[row]]), dim=0)
            for row in range(len(extents))
        ],
        dim=0,
    )

    model = MultiObstacleHallucinator(stations, obstacles=obstacles)
    if seed_latents:
        model.seed(np.mean(np.stack(seed_latents), axis=0))
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sharp_temperature = decoder.temperature_m
    history: list[dict] = []
    loss_value = reconstruction_value = float("nan")
    for step in range(steps):
        if anneal_from_m is not None and steps > 1:
            # Geometric schedule from soft to the configured sharpness.
            progress = step / (steps - 1)
            decoder.temperature_m = float(
                anneal_from_m * (sharp_temperature / anneal_from_m) ** progress
            )
        # Plausibility is bought after the mechanism works: zero for the first third, then
        # ramped in, so the prior tightens around a solution rather than preventing one.
        shape_ramp = 0.0
        if anneal_prior and steps > 1:
            share = step / (steps - 1)
            shape_ramp = float(np.clip((share - 0.33) / 0.5, 0.0, 1.0))
        mean, log_sigma = model(profiles)
        sigma = log_sigma.exp()
        reconstruction = torch.zeros((), dtype=torch.float32)
        clearance = torch.zeros((), dtype=torch.float32)
        repulsion = torch.zeros((), dtype=torch.float32)
        easy_term = torch.zeros((), dtype=torch.float32)
        minimality = torch.zeros((), dtype=torch.float32)
        shape_term = torch.zeros((), dtype=torch.float32)
        for _ in range(samples):
            latent = mean + sigma * torch.randn_like(mean)
            for row in range(len(extents)):
                boxes = geometry.decode(latent[row])
                if shape_ramp:
                    shape_term = shape_term + shape_ramp * geometry.size_penalty(
                        boxes, shape or PlausibleShape()
                    )
                weights = decoder(extents[row], boxes, costs[row])
                reconstruction = reconstruction - torch.log(weights[observed[row]] + 1e-8)
                # Easy scene: with the obstacles moved clear, the cheapest candidate -- the
                # nominal -- must be preferred again. This is the other half of the 2x2.
                easy_weights = decoder(extents[row], relax(boxes), costs[row])
                easy_term = easy_term - torch.log(easy_weights[0] + 1e-8)
                # Minimality: the scene should sit just tight enough to force the edit, not
                # bury the nominal. A face driven far past the nominal is a scene the observed
                # motion did not need, which is regret expressed as a loss.
                blocked = decoder.penetration(extents[row], boxes)
                minimality = minimality + blocked[0].clamp(min=0.0).pow(2)
                # The observed motion must remain executable: no obstacle may sit on it.
                observed_extents = extents[row][observed[row] : observed[row] + 1]
                intrusion = decoder.penetration(observed_extents, boxes)
                clearance = clearance + torch.relu(intrusion + clearance_m).pow(2).mean()
                # Obstacles must not pile onto each other, as in LfLH's obstacle-obstacle term.
                station_gap = (boxes["station"][:, None] - boxes["station"][None, :]).abs()
                lateral_gap = (boxes["lateral_m"][:, None] - boxes["lateral_m"][None, :]).abs()
                mask = 1.0 - torch.eye(obstacles, dtype=torch.float32)
                overlap = torch.relu(1.0 - station_gap) * torch.relu(0.3 - lateral_gap)
                repulsion = repulsion + (overlap * mask).mean()
        scale = samples * len(extents)
        reconstruction = reconstruction / scale
        loss = (
            reconstruction
            + easy_weight * easy_term / scale
            + minimality_weight * minimality / scale
            + shape_weight * shape_term / scale
            + kl_weight * _kl(mean, log_sigma)
            + clearance_weight * clearance / scale
            + repulsion_weight * repulsion / scale
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_value = float(loss.detach())
        reconstruction_value = float(reconstruction.detach())
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "reconstruction": reconstruction_value,
                    "mean_sigma": float(sigma.mean().detach()),
                }
            )

    decoder.temperature_m = sharp_temperature
    with torch.no_grad():
        mean, log_sigma = model(profiles)
    return model, TrainingReport(
        steps=steps,
        final_loss=loss_value,
        reconstruction=reconstruction_value,
        mean_sigma=float(log_sigma.exp().mean()),
        history=history,
    )


def sample_scenes(
    model: MultiObstacleHallucinator,
    extents: np.ndarray,
    costs: np.ndarray,
    observed_index: int,
    motion_id: str,
    *,
    count: int,
    geometry: ObstacleGeometry | None = None,
    decoder: ChoiceDecoder | None = None,
    seed: int = 0,
) -> SceneSampleBatch:
    """Draw obstacle sets and record which candidate each one selects."""
    torch.manual_seed(seed)
    stations = extents.shape[-1]
    geometry = geometry or ObstacleGeometry(stations=stations)
    decoder = decoder or ChoiceDecoder()
    extent_tensor = torch.tensor(extents, dtype=torch.float32)
    cost_tensor = torch.tensor(costs, dtype=torch.float32)
    profile = torch.stack((extent_tensor[0], extent_tensor[observed_index]), dim=0)[None, ...]
    with torch.no_grad():
        mean, log_sigma = model(profile)
        sigma = log_sigma.exp()
        rows, winners = [], []
        for _ in range(count):
            latent = mean[0] + sigma[0] * torch.randn_like(mean[0])
            boxes = geometry.decode(latent)
            weights = decoder(extent_tensor, boxes, cost_tensor)
            winners.append(int(weights.argmax()))
            rows.append(
                torch.stack(
                    (
                        boxes["station"],
                        boxes["lateral_m"],
                        boxes["height_m"],
                        boxes["half_along_m"],
                        boxes["half_lateral_m"],
                        boxes["half_vertical_m"],
                    ),
                    dim=-1,
                ).numpy()
            )
    return SceneSampleBatch(
        motion_id=motion_id,
        parameters=np.stack(rows, axis=0),
        winner=np.asarray(winners),
        observed_index=observed_index,
    )
