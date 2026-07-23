# verl integration for the MaxRL repo

Files to integrate the curriculum teacher into the official MaxRL codebase
(https://github.com/tajwarfahim/maxrl, a verl fork):

- `curriculum.py` — copy to `verl/utils/curriculum.py`. FrontierTeacher
  (Beta-posterior + Thompson sampling over the derived advantage-mass utility)
  and CurriculumSampler (weighted sampler, re-draws weights each epoch).
- `main_ppo.patch` — enables the sampler behind `+data.curriculum.enable=true`.
- `ray_trainer.patch` — observe hook after reward computation, wandb metrics,
  teacher state persisted/restored with checkpoints.
- `smollm_curriculum.sh` — SmolLM2-360M + GSM8K launch script (fill in paths).

Apply from the MaxRL repo root:

    cp curriculum.py <maxrl>/verl/utils/curriculum.py
    cd <maxrl> && git apply main_ppo.patch ray_trainer.patch

Requires the dataset preprocessor to store the row index in
`extra_info.index` (the repo's gsm8k.py already does).
