# Reproduce the released selectors on CPU

Keep both [`selectors.tar.gz`](selectors.tar.gz) and
[`selectors-manifest.json`](selectors-manifest.json) in the same directory. The
verification commands read the compressed archive directly; extracting its inner
`manifest.json` does not replace the external archive manifest.

Extract the [matching research source](../independent-layout-source-20260906/research-source.tar.gz)
and run from that source root with NumPy and PyTorch. Package versions are recorded
in `selectors-manifest.json` and the source manifest.

```bash
PYTHONPATH=. python scripts/research/bundle_motion2scene_selectors.py verify --out /path/to/bundle
```

This reproduces the twenty existing training fits and writes a new
`selectors-reproduction.json`; use a directory without an existing receipt.
The [published receipt](selectors-reproduction.json) records the measured exact
agreement in the recorded environment.

To reproduce the independent-layout readouts, also download
[`independent-layout-decisions.json`](../independent-layout-decisions.json):

```bash
PYTHONPATH=. python scripts/research/verify_motion2scene_layout_readouts.py \
  --bundle /path/to/bundle --decisions /path/to/independent-layout-decisions.json
```

These commands require no simulator and perform no physical rollouts. The latter
checks the original requests, refusals and probabilities, not an unexecuted
alternative's outcome. SONIC, motion banks and simulator assets are external.
