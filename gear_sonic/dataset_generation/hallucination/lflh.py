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
        self.encoder = nn.Sequential(
            nn.Conv1d(3, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * obstacles * OBSTACLE_PARAMS),
        )

    def forward(self, profiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(profiles)
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=-1)
        raw = self.head(pooled)
        batch = raw.shape[0]
        raw = raw.view(batch, self.obstacles, 2 * OBSTACLE_PARAMS)
        mean = raw[..., :OBSTACLE_PARAMS]
        log_sigma = raw[..., OBSTACLE_PARAMS:].clamp(min=self.min_log_sigma, max=2.0)
        return mean, log_sigma


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
    min_half_extent_m: float = 0.04
    max_half_extent_m: float = 0.80

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        station = torch.sigmoid(latent[..., 0]) * (self.stations - 1)
        lateral = torch.tanh(latent[..., 1]) * self.lateral_scale_m
        height = self.height_centre_m + torch.tanh(latent[..., 2]) * self.height_scale_m
        extents = self.min_half_extent_m + (
            self.max_half_extent_m - self.min_half_extent_m
        ) * torch.sigmoid(latent[..., 3:6])
        return {
            "station": station,
            "lateral_m": lateral,
            "height_m": height,
            "half_along_m": extents[..., 0],
            "half_lateral_m": extents[..., 1],
            "half_vertical_m": extents[..., 2],
        }


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
    clearance_weight: float = 1.0,
    repulsion_weight: float = 0.2,
    clearance_m: float = 0.05,
    seed: int = 0,
    geometry: ObstacleGeometry | None = None,
    decoder: ChoiceDecoder | None = None,
) -> tuple[MultiObstacleHallucinator, TrainingReport]:
    """LfLH's objective: reconstruct the observed choice, with size KL, clearance and repulsion."""
    torch.manual_seed(seed)
    stations = batch_extents[0].shape[-1]
    geometry = geometry or ObstacleGeometry(stations=stations)
    decoder = decoder or ChoiceDecoder()
    extents = [torch.tensor(item, dtype=torch.float32) for item in batch_extents]
    costs = [torch.tensor(item, dtype=torch.float32) for item in batch_costs]
    # The hallucinator sees only the observed motion, as LfLH sees only the executed plan.
    profiles = torch.stack([extents[row][observed[row]] for row in range(len(extents))], dim=0)

    model = MultiObstacleHallucinator(stations, obstacles=obstacles)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    loss_value = reconstruction_value = float("nan")
    for step in range(steps):
        mean, log_sigma = model(profiles)
        sigma = log_sigma.exp()
        reconstruction = torch.zeros((), dtype=torch.float32)
        clearance = torch.zeros((), dtype=torch.float32)
        repulsion = torch.zeros((), dtype=torch.float32)
        for _ in range(samples):
            latent = mean + sigma * torch.randn_like(mean)
            for row in range(len(extents)):
                boxes = geometry.decode(latent[row])
                weights = decoder(extents[row], boxes, costs[row])
                reconstruction = reconstruction - torch.log(weights[observed[row]] + 1e-8)
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
    profile = extent_tensor[observed_index][None, ...]
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
