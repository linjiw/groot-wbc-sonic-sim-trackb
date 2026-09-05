# Motion2Scene visual research report

This static GitHub Pages report covers the fresh-machine shared-seed Kimodo motion qualification
study as of 2026-09-05. It does not pool outcomes with the earlier SweepCF or parallel learned
geometry-only hallucination track.

## Rebuild

Run from the repository root with the trusted local research bundle and standalone Motion2Scene
research repository available:

```bash
MUJOCO_GL=egl .venv_research/bin/python scripts/research/render_motion2scene_report.py \
  --data-root /path/to/research-data/groot-wbc \
  --research-repo /path/to/motion2scene
```

Dependencies: NumPy, MuJoCo, Matplotlib, Pillow, ImageIO/FFmpeg, and the repository's G1 meshes.
`--preview` renders only midpoint poster frames. Full generation produces three MP4 replays,
three posters, three SVG figures, portable evidence snapshots, and a hash manifest.

The first two videos replay recorded Isaac states. The third replays shared-clock reference CSVs
with a manually placed illustrative beam. No `mj_step` is called; none of these renders creates
a physics verdict. Existing recorder metadata supplies the verdicts for the first two videos.
The beam scene is not an admitted critical scene and must not be represented as a physical
preference-reversal result. Videos use a shared camera and elapsed time across panels.

`assets/manifest.json` records the selected source hashes and output hashes. Original absolute
machine paths are shortened in public JSON copies; source hashes and public snapshot hashes are
therefore separate. The full non-git motion bundle is needed to rebuild the media.

## Validation

```bash
.venv_research/bin/ruff check --select E,F,I scripts/research/render_motion2scene_report.py
.venv_research/bin/black --check scripts/research/render_motion2scene_report.py
git diff --check
.venv_research/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Browser validation: Chrome headless via Playwright, 1440×1000 desktop and 390×844 mobile;
no JavaScript errors or horizontal overflow. All three video sources decode and play; durations
are 3.967, 3.967 and 4.767 seconds, respectively. The synthetic beam slider is tested at both
endpoints (no feasible motion / neutral feasible). Public links, evidence hashes and media
metadata are checked before publication. No new controller experiment is performed for this report.
