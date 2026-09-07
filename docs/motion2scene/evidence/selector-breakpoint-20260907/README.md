# Selector breakpoint: downloadable evidence and CPU reproduction

Download [breakpoint-evidence.tar.gz](breakpoint-evidence.tar.gz) and its
[manifest](breakpoint-evidence-manifest.json), then extract the archive into an empty
directory. It includes the four linear weights, deployed containers, full diagnostic
records, two retained numerical failures, all 36 physical outcomes and export hashes.
The force and trajectory artifacts referenced inside records remain external;
this evidence archive is not a complete simulator installation or raw-motion release.

Also download the original [selector bundle](../selector-bundle-20260906/README.md)
and extract the [matching source archive](../selector-breakpoint-source-20260907/research-source.tar.gz).
From that source root, with the recorded Python dependencies installed:

```bash
PYTHONPATH=. python scripts/research/verify_motion2scene_linear_controls.py \
  --bundle /path/to/original-selector-bundle \
  --diagnostic /path/to/extracted-breakpoint-evidence \
  --output /tmp/new-linear-reproduction.json
```

The verifier reads the original compressed training corpus directly, checks exported
file hashes, refits all four controls from their original eleven-example subsets,
and compares their weights/scales and all 24 deployed decisions. Its output must be
a new file. The [published reproduction receipt](linear-reproduction.json) reports
the measured agreement in the recorded environment. No simulator or new physical
outcome is produced by this command.

P0 has twenty original frozen MLP diagnostics and four new controls; P1 has twelve
new paired command runs; P2 is a recorded empty-transition forecast; P3 has twenty-four
new executions of the common linear controls. These are inspected development
conditions on one source. The original evaluation stays 120/600 admitted with 480
paused assignments. Refusal is neutral commitment, not physical stopping. The
analytic rescue does not establish Motion2Scene's training-data advantage.
