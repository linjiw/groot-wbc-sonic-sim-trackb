# Interface v1 resource continuation

Registered 2026-09-06 after the first six cells completed and the second seed
was not started because free GPU memory was below the 9000 MiB startup floor.
The shared GPU fluctuates around 7–9 GiB free. Do not terminate other workloads.

Continue only the six unstarted cells with the previously validated 7500 MiB
trajectory-only floor used by the completed source-execution and d040 batches.
This interface adds small ray-query and force buffers and no RTX camera. Keep
serial execution, the 375 s timeout, and all science/controller settings unchanged.
The original six completed cells are revalidated and reused, never rerun. Preserve
the yielded manifest/record via hashes in a new directory. Carry original charged
GPU time once into the consolidated record; charge only new launches thereafter.
Retain every sensor failure. The complete 12-cell P1/P2/P3 evaluation remains the
original registered experiment; this is a disclosed scheduling exception.
