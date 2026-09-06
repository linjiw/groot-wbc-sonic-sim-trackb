# Beam intervention: trajectory-only resource amendment

2026-09-06, before the first beam-intervention physics run. The user explicitly
requested proceeding with the available VRAM. Measurements show 8.4–8.7 GiB free,
below the conservative 9,000 MiB threshold but above the previously used trajectory-only
7,500 MiB guard documented in LADDER_EXTENSION_RESOURCE_V2.md. That record cites
completed trajectory-only runs using less than 4.9 GiB. The beam adds a single rigid
body and one filtered contact sensor; its actual runtime memory remains to be measured.

Create a new resource-v3 directory and preserve all earlier manifests/run records.
Change the free-memory launch floor to **7,500 MiB**, retaining serial execution,
trajectory-only capture, 375-second timeout, infrastructure stop rules, and the
existing 1.4583 GPU-hour ceiling for two controls plus twelve paired runs. Never stop
another workload or retry a scientific rejection. Record infrastructure failures
separately and retain any artifacts they produce.

All beam poses, collision settings, motions, checkpoint, physics seeds, passage and
force thresholds remain those in COUNTERFACTUAL_STAGE_V1.md. No prior scientific
beam-intervention outcome exists. Bind the original scientific protocol, this resource
amendment, and the previous prelaunch manifests/run records by hash. The resource
change supplies no scientific qualification or guarantee of sufficient runtime memory.
