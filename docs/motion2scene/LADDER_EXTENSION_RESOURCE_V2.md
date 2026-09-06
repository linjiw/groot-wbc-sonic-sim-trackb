# Intermediate-level extension: resource-only amendment

Registered before any extension physics, 2026-09-05. The v1 extension manifest yielded repeatedly
below its 9000 MiB startup threshold and launched zero cells. Preserve that manifest and run record.
The new resource-v2 lineage changes the startup threshold to **7500 MiB**, uses a fresh output
directory and records this amendment as its prediction document.

All three scientific predictions, motion/reference identities, physics seeds 7901–7903,
comparators, scoring code, serial trajectory-only mode and 375-second timeout remain exactly
those in [LADDER_EXTENSION_V1.md](LADDER_EXTENSION_V1.md). No scientific result existed when this
amendment was made. No tracker, route, geometry or behavior threshold changes.

This restores the trajectory-only resource contract already documented in the standalone
`experiments/registrations/E0_Q4_PROTOCOL_V1.json`: completed runs used less than 4.9 GiB, with
a 7500 MiB startup guard. That contract is resource evidence only; this extension is not a Q4
experiment and does not change Q4 eligibility. The old held-out route experiment likewise used
7500 MiB. This guard is not a reservation; the runner still yields below it, stops on infrastructure
failure, retains scientific rejections, and never stops another workload.

The projected ceiling remains 0.3125 contended GPU-hours. The amended manifest records the
original manifest and prelaunch run-record hashes, and binds the unchanged original prediction
document. Changed fields are resource floor, output paths, experiment/resource note, registered
prediction pointer and amendment provenance. The three cells and scientific analysis are unchanged.
