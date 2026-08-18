#!/usr/bin/env python3
"""Render one counterfactual family as a page that can be checked by eye.

The numbers say the family holds. A reader still has to be able to see it, and the failure
modes this pipeline has actually produced were all visible ones: an adapted motion that
walked into a wall, a shelf placed where the duck had not yet begun, a rejection that came
from drift rather than from the obstacle. Each was found by looking, after the numbers had
already said everything was fine.

So the page carries what a reviewer needs to disbelieve it:

* the 2x2, with the force and the contacting body in each cell
* a side elevation of both scenes against both motions' silhouette profiles, which is where
  the window either visibly exists or does not
* predicted against observed first contact, the one number that ties geometry to physics
* ego keyframes either side of the contact frame
* the claim level, which is never asserted -- it is read from the artefacts that exist

That last point matters. A family is "perturbation robust" only if robustness.json says so,
and absent that file the page says "nominally physics verified" rather than leaving the
stronger claim implied.

Usage::

    python scripts/research/build_family_report.py --family /data/.../duck_003
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gear_sonic.dataset_generation.motion_envelope import (  # noqa: E402
    silhouette_at_stations,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

CELLS = ("nominal_easy", "nominal_hard", "adapted_easy", "adapted_hard")


def load(directory: Path) -> dict | None:
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        return None
    with paths[0].open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    return payload


def as_data_uri(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def elevation_figure(probes: dict, family: dict) -> str:
    """Side view: both silhouette profiles, both shelf heights, the window between them."""
    figure, axes = plt.subplots(figsize=(9, 4.2))
    ranges = [
        (float(np.asarray(p["root_pos_w"])[:, 0].min()),
         float(np.asarray(p["root_pos_w"])[:, 0].max()))
        for p in probes.values()
    ]
    low = max(r[0] for r in ranges)
    high = min(r[1] for r in ranges)
    stations = np.arange(low, high, 0.05)

    for label, colour in (("nominal", "#c1442e"), ("adapted", "#2e6fc1")):
        peaks = silhouette_at_stations(probes[label], stations)
        axes.plot(stations, peaks, color=colour, linewidth=2.0,
                  label=f"{label} silhouette peak")

    for key, style, colour in (
        ("easy_shelf_underside_m", "--", "#7a9e5a"),
        ("hard_shelf_underside_m", "-", "#b5852a"),
    ):
        if key in family:
            axes.axhline(family[key], linestyle=style, color=colour, linewidth=1.6,
                         label=f"{key.split('_')[0]} shelf {family[key]:.3f} m")

    if "station_x_m" in family and family["station_x_m"] is not None:
        axes.axvline(family["station_x_m"], color="#888", linewidth=1.0, linestyle=":")

    axes.set_xlabel("position along the route, x (m)")
    axes.set_ylabel("highest point of the robot (m)")
    axes.set_title("Where the two motions differ, and where each shelf sits")
    axes.legend(fontsize=8, loc="lower left")
    axes.grid(alpha=0.25)
    return as_data_uri(figure)


def plan_figure(probes: dict) -> str:
    """Top view: both executed routes, to show they are doing the same task."""
    figure, axes = plt.subplots(figsize=(9, 3.0))
    for label, colour in (("nominal", "#c1442e"), ("adapted", "#2e6fc1")):
        root = np.asarray(probes[label]["root_pos_w"])
        axes.plot(root[:, 0], root[:, 1], color=colour, linewidth=2.0, label=label)
        axes.plot(root[0, 0], root[0, 1], "o", color=colour, markersize=5)
        axes.plot(root[-1, 0], root[-1, 1], "s", color=colour, markersize=5)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_title("Executed routes: circle is the start, square the goal")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.25)
    axes.set_aspect("equal", adjustable="datalim")
    return as_data_uri(figure)


def keyframes(directory: Path, frame: int | None, count: int = 3) -> list[str]:
    """Ego frames around the contact, pulled straight out of the recorded video."""
    videos = sorted(directory.glob("renders/*.mp4"))
    if not videos or frame is None:
        return []
    images = []
    for offset in range(-1, count - 1):
        target = max(0, frame + offset * 8)
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(videos[0]),
             "-vf", f"select=eq(n\\,{target})", "-vframes", "1", "-f", "image2pipe",
             "-vcodec", "png", "-"],
            capture_output=True, check=False,
        )
        if result.returncode == 0 and result.stdout:
            images.append(
                "data:image/png;base64," + base64.b64encode(result.stdout).decode()
            )
    return images


def claim_level(family_dir: Path) -> tuple[str, str]:
    """Read the claim from the artefacts, never assert it."""
    attribution = family_dir / "attribution.json"
    robustness = family_dir / "robustness.json"
    if robustness.exists():
        report = json.loads(robustness.read_text())
        if report.get("perturbation_robust"):
            return "perturbation robust", "strong"
        return (
            "nominally physics verified (perturbation test ran and did not hold)", "weak"
        )
    if attribution.exists() and json.loads(attribution.read_text()).get("attribution_pure"):
        return "nominally physics verified (not yet perturbed)", "medium"
    return "geometrically predicted only", "weak"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    family = json.loads((args.family / "family.json").read_text())
    attribution = {}
    if (args.family / "attribution.json").exists():
        attribution = json.loads((args.family / "attribution.json").read_text())

    probes = {label: load(args.family / f"probe_{label}") for label in ("nominal", "adapted")}
    if any(payload is None for payload in probes.values()):
        raise SystemExit("probe trajectories missing; cannot draw the geometry")

    level, strength = claim_level(args.family)
    contact_frame = (
        attribution.get("cells", {}).get("nominal_hard", {}).get("first_contact_frame")
    )

    rows = []
    for cell in CELLS:
        info = attribution.get("cells", {}).get(cell, {})
        result = family.get("results", {}).get(cell, {})
        outcome = info.get("outcome") or result.get("outcome", "-")
        body = info.get("first_contact_body") or ""
        frame = info.get("first_contact_frame")
        contact = (
            f"{body} @ frame {frame}, {info.get('first_contact_force_n', 0):.1f} N"
            if frame is not None else "no lateral contact"
        )
        rows.append((cell, outcome, contact, info.get("predicted_clearance_m")))

    images = keyframes(args.family / "nominal_hard", contact_frame)
    elevation = elevation_figure(probes, family)
    plan = plan_figure(probes)

    def cell_html(cell, outcome, contact, clearance):
        css = "pass" if outcome == "accepted" else "fail"
        clear = f"{clearance:+.4f} m" if isinstance(clearance, (int, float)) else "-"
        return (
            f'<tr><td class="k">{cell}</td><td class="{css}">{outcome}</td>'
            f"<td>{contact}</td><td class=num>{clear}</td></tr>"
        )

    predicted = attribution.get("cells", {}).get("nominal_hard", {})
    agreement = ""
    if predicted.get("first_contact_frame") is not None and \
            predicted.get("predicted_first_interference_frame") is not None:
        observed = predicted["first_contact_frame"]
        expected = predicted["predicted_first_interference_frame"]
        agreement = (
            f"<p class=head>Observed first contact at frame <b>{observed}</b>; "
            f"swept-volume geometry predicted first interference at frame "
            f"<b>{expected}</b> &mdash; {abs(observed - expected)} frame(s) apart, "
            f"{abs(observed - expected) / 50 * 1000:.0f} ms.</p>"
        )

    html = f"""<title>{family['family_id']}</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#23201c; --mut:#6b645b; --line:#ddd8cf; --pass:#2f6b3a; --fail:#a8392b; }}
:root:not([data-theme="light"]) {{ }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#191714; --fg:#ece7df; --mut:#a49c90; --line:#3a352e; --pass:#7fc08c; --fail:#e2897b; }} }}
:root[data-theme="dark"] {{ --bg:#191714; --fg:#ece7df; --mut:#a49c90; --line:#3a352e;
  --pass:#7fc08c; --fail:#e2897b; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 Georgia,serif;
  max-width:60rem; margin:0 auto; padding:2.5rem 1.5rem; }}
h1 {{ font-size:1.7rem; margin:0 0 .2rem; }}
.sub {{ color:var(--mut); margin:0 0 1.6rem; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0 1.6rem; font-size:14px; }}
th,td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line); }}
th {{ font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--mut); }}
.k {{ font-family:ui-monospace,monospace; }}
.num {{ font-variant-numeric:tabular-nums; text-align:right; }}
.pass {{ color:var(--pass); font-weight:600; }}
.fail {{ color:var(--fail); font-weight:600; }}
.badge {{ display:inline-block; padding:.3rem .7rem; border:1px solid var(--line);
  border-radius:2rem; font-size:13px; }}
.strong {{ border-color:var(--pass); color:var(--pass); }}
.medium {{ color:var(--mut); }}
.weak {{ border-color:var(--fail); color:var(--fail); }}
figure {{ margin:1.4rem 0; }} img {{ max-width:100%; }}
.frames {{ display:flex; gap:.5rem; overflow-x:auto; }} .frames img {{ height:150px; }}
.head {{ background:rgba(128,128,128,.09); padding:.7rem .9rem; border-radius:4px; }}
</style>
<h1>{family['family_id']}</h1>
<p class=sub>{family.get('regime','overhead')} regime &middot; window
{family.get('window_m', 0):.3f} m &middot; nominal clears to
{family.get('nominal_clears_to_m', float('nan')):.3f} m, adapted to
{family.get('adapted_clears_to_m', float('nan')):.3f} m</p>

<p><span class="badge {strength}">claim level: {level}</span></p>

<h2>The 2&times;2</h2>
<table><tr><th>cell</th><th>outcome</th><th>first lateral contact</th>
<th>predicted clearance</th></tr>
{''.join(cell_html(*row) for row in rows)}
</table>
{agreement}

<h2>Where the window is</h2>
<figure><img src="{elevation}" alt="silhouette profiles against both shelf heights"></figure>
<figure><img src="{plan}" alt="executed routes"></figure>

<h2>Ego view around the contact</h2>
{'<div class=frames>' + ''.join(f'<img src="{src}">' for src in images) + '</div>'
 if images else '<p class=sub>No ego frames available for the failing cell.</p>'}
"""
    destination = args.out or args.family / "report.html"
    destination.write_text(html, encoding="utf-8")
    print(f"claim level: {level}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
