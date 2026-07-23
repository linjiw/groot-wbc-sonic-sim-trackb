# GSM8K x Curriculum-MaxRL on a single A10G — feasibility plan

Investigated 2026-07-22 on this box (A10G 23GB / 30GB RAM / 8 cores, Amazon Linux 2023).
Target: scaled-down replica of the paper's SmolLM2 + GSM8K recipe (`smollm/smollm_curriculum.sh`,
8 GPUs, vllm, 256 prompts x 128 rollouts) as a 2x2 ablation (curriculum on/off x maxrl/grpo),
each cell <= 8 h, using the **hf rollout path** (`actor_rollout_ref.rollout.name=hf`) that the
paper itself uses for small models in `maze/maze_17.sh`.

**Verdict: GO — but not on python3.9.** Use a `/usr/bin/python3.11` venv (present on this box).
Details and the reason below.

---

## 1. Dependency gap analysis (import checks actually run)

### What is already present (system python3.9, `pip3 --user`)

| present | version |
|---|---|
| torch | 2.6.0+cu124 (CXX11 ABI = False) |
| numpy | 2.0.2 |
| triton | 3.2.0 |
| sympy | 1.13.1 |
| filelock, fsspec, requests, pyyaml, packaging | ok |

### What is missing (all verified `ModuleNotFoundError`)

ray, pandas, transformers, hydra, omegaconf, tensordict, peft, codetiming, torchdata,
datasets, pyarrow, vllm, flash_attn, wandb, accelerate, pylatexenc, dill, psutil,
safetensors, tqdm, math_verify.

### Two hard findings that shape the install

1. **`math-verify` cannot run on python3.9 — verified, not just metadata.** Every version
   (0.3.3, 0.5.2, 0.9.0 tested by unpacking wheels) crashes at import with
   `TypeError: unsupported operand type(s) for |` from *runtime-evaluated* PEP-604 unions
   (`ExtractionTarget = LatexExtractionConfig | ExprExtractionConfig | ...` in `parser.py`,
   same in `latex2sympy2_extended/symbols.py`). A `from __future__ import annotations`
   patch does NOT fix it (the unions are module-level expressions, not annotations).
   And math-verify is a *hard* import of the trainer: `verl/workers/reward_manager/__init__.py`
   unconditionally imports `multi_thread_naive.py` and `naive.py`, both of which do
   `from math_verify import verify` at module top. So python3.9 is dead for this repo
   without code surgery. On math-verify 0.5.2+antlr I confirmed the scorer itself works
   fine once the interpreter is >= 3.10 (`\boxed{72}` vs `72` -> 1.0).
2. **`flash_attn` is a hard import too** (not just an attn backend): `verl/workers/actor/dp_actor.py`
   and `verl/workers/critic/dp_critic.py` do `if is_cuda_available: from flash_attn.bert_padding import ...`
   at module scope, and `workers/actor/__init__.py` imports dp_actor. No compile needed though:
   a prebuilt wheel exists for exactly our stack —
   `flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl` (256 MB,
   torch 2.6 / CUDA 12 / cxx11abi FALSE matches our torch). This also gives us the fast
   fused cross-entropy in `logprobs_from_logits`.

Also verified: `/usr/bin/python3.11` (3.11.15) exists with working `venv`/`ensurepip`;
`torch-2.6.0+cu124` cp311 wheels exist on download.pytorch.org; PyPI and huggingface.co
are both reachable (HTTP 200). `uv` 0.11.26 is available if preferred.

One more repo-level py3.9 incompatibility found while compiling the tree:
`verl/tools/utils/tool_registry.py` uses `match` (py>=3.10). It is only imported by the
sglang rollout, so it never loads on our path — but it confirms this fork is not 3.9-clean.

### Exact install (python3.11 venv, no conda)

```bash
/usr/bin/python3.11 -m venv ~/venvs/maxrl311
source ~/venvs/maxrl311/bin/activate
pip install -U pip

# 1. torch matching the existing CUDA userland
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. prebuilt flash-attn (no nvcc build; matches torch2.6/cu12/cxx11abiFALSE)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# 3. verl runtime deps for the ray+fsdp+hf-rollout path (NO vllm, NO sglang, NO megatron)
pip install "ray[default]==2.46.0" transformers==4.51.3 tensordict==0.6.2 \
    torchdata==0.11.0 datasets==3.6.0 hydra-core==1.3.2 "omegaconf>=2.3,<2.4" \
    accelerate peft codetiming pandas pyarrow einops safetensors psutil tqdm \
    math-verify==0.8.0 pylatexenc
# optional: wandb  (plan below uses trainer.logger=['console'] to avoid it)

cd ~/work/curriculumrl/maxrl && pip install -e . --no-deps
```

Notes on pins: `tensordict==0.6.2` is what this fork's dockerfiles pin (`<=0.6.2`);
`torchdata==0.11.0` provides `torchdata.stateful_dataloader.StatefulDataLoader` (needed by
`ray_trainer.py:36`); `hydra-core 1.3.2` pins `antlr4-python3-runtime==4.9.*` which
intersects math-verify's `>=4.9.3,<=4.13.2` at **4.9.3** — resolvable, no conflict.
vllm/sglang/megatron imports in `fsdp_workers.py` are inside `rollout_name == "vllm"` etc.
branches; the `"hf"` branch (line ~405) only needs `HFRollout` + `BaseShardingManager` — verified.

---

## 2. Memory budget (single-GPU FSDP + hf rollout, bf16 compute / fp32 master)

Model shapes (fetched from HF configs, both llama-type, tied embeddings, head_dim 64):

| | 360M-Instruct | 135M-Instruct |
|---|---|---|
| hidden / layers / heads / kv-heads | 960 / 32 / 15 / 5 | 576 / 30 / 9 / 3 |
| params (computed) | ~362 M | ~134 M |
| KV-cache bytes/token (bf16) | 40 KB | 23 KB |

verl's actor keeps **fp32 master weights** by default (`fsdp_workers.py:210`,
`torch_dtype = float32 if self._is_actor`), with bf16 `MixedPrecision(param_dtype)` for compute.
On 1 GPU FSDP is a no-op shard, so everything is resident.

Peak VRAM per phase, 360M, prompt 512 + response 1024 (=1536 max seq):

| component | 360M | 135M |
|---|---|---|
| fp32 params + grads + AdamW (16 B/param) | 5.8 GB | 2.1 GB |
| bf16 gathered copy during rollout (`summon_full_params`) | +1.5 GB | +0.6 GB |
| rollout KV cache @ **micro_batch=64 samples** x 1536 tok | 3.9 GB | 2.3 GB |
| old_log_prob logits @ **micro=16** x 1536 x 49152 (bf16, fused CE) | 2.4 GB | 2.4 GB |
| update: logits+grad @ **micro=8** x 1536 (grad ckpt on) | ~3.0 GB | ~3.0 GB |
| **worst-phase total** | **~11–12 GB** | **~7–8 GB** |

Comfortably inside 22 GiB usable, with headroom for HF-generate scratch and fragmentation.
Recommended micro sizes (all sample-level):

- 360M: `+actor_rollout_ref.rollout.micro_batch_size=64`, `log_prob_micro_batch_size_per_gpu=16`,
  `ppo_micro_batch_size_per_gpu=8`
- 135M: micro_batch=128, log_prob 32, ppo micro 16 (2x everything)

Two memory-safety switches worth setting: `actor_rollout_ref.actor.entropy_from_logits_with_chunking=True`
(entropy is computed during old_log_prob with `calculate_entropy=True`; the unchunked path can
materialize fp32-vocab tensors) and keep `enable_gradient_checkpointing: true` (default).

---

## 3. Concrete scaled-down config

**Scaling logic.** Paper: 256 prompts x 128 rollouts = 32,768 seqs/step over 8 GPUs
(4,096/GPU). Ours: 64 prompts x 16 rollouts = 1,024 seqs/step on 1 GPU — a 4x/GPU reduction
that offsets hf-generate being ~4-10x slower than vllm. N=16 keeps enough group resolution
for MaxRL (truncate order T=N=16) and for the teacher's `pass@N - pass@1` utility.
`max_response_length=1024` instead of 2048: SmolLM2 GSM8K solutions are ~150-400 tokens;
2048 doubles KV + straggler cost for negligible pass-rate gain, and truncated responses are
already zeroed by `zero_reward_on_max_response_length`.

**Throughput estimate (A10G).** HF `generate` on a 360M model is kernel-launch/Python bound:
~15-30 ms/decode step regardless of batch<=64 -> ~2,000-4,000 tok/s aggregate at 64 concurrent
sequences. Per training step: 16 chunks x ~400-700 steps x ~20 ms ~= **3-6 min generation**;
old_log_prob ~0.5-1 min (0.43 PFLOP); update ~1.5-3 min (1.2 PFLOP with grad ckpt);
reward ~10-20 s (1,024 math-verify calls on 4 actors). **~6-10 min/step -> 50 steps ~= 5-8 h.**
135M: ~2.5-3x faster, ~3 min/step, 100 steps ~= 5 h.

**Recommendation: 360M, 50 steps, per cell <= 8 h** (135M x 100 steps is the fallback if the
first cell overruns; same script, one variable change). 2x2 grid = 4 sequential runs ~= 24-32 h.

### `smollm/smollm_a10g.sh` (proposed; overrides in the house style)

```bash
MODEL_PATH=HuggingFaceTB/SmolLM2-360M-Instruct     # HF id works: copy_to_local passes non-hdfs paths through to transformers
TRAIN_DATA=$HOME/data/gsm8k/train.parquet
VAL_DATA=$HOME/data/gsm8k/test_256.parquet          # 256-row slice of platinum, see §4
ADVANTAGE_ESTIMATOR=maxrl                           # or grpo
CURRICULUM=true                                     # or false
N_ROLLOUTS=16

# NOTE: do NOT `ray start` externally; main_ppo self-inits (nnodes=1 -> address=None).

python3 -m verl.trainer.main_ppo \
  ray_init.ray_dir=/tmp/ray \
  ray_init.num_cpus=32 \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.pass_k=${N_ROLLOUTS} \
  algorithm.truncate_order=${N_ROLLOUTS} \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${VAL_DATA} \
  data.train_batch_size=64 \
  data.filter_overlong_prompts=True \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  +data.dataloader_num_workers=2 \
  +data.curriculum.enable=${CURRICULUM} \
  +data.curriculum.floor=0.1 \
  +data.curriculum.decay=0.9 \
  +data.curriculum.success_threshold=0.5 \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.micro_batch_size=64 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=0 \
  reward_model.reward_manager=multi_thread \
  +reward_model.reward_kwargs.num_reward_actors=4 \
  +reward_model.reward_kwargs.zero_reward_on_max_response_length=True \
  +reward_model.reward_kwargs.max_resp_len=1024 \
  trainer.project_name=CurriculumMaxRL_SmolLM_A10G \
  trainer.experiment_name=${ADVANTAGE_ESTIMATOR}_cur${CURRICULUM}_${N_ROLLOUTS}r \
  trainer.logger=['console'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=200 \
  trainer.total_training_steps=50 \
  trainer.save_freq=25 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=25 \
  trainer.default_local_dir=$HOME/ckpt/gsm8k_a10g/${ADVANTAGE_ESTIMATOR}_cur${CURRICULUM}
```

Key deltas vs the 8-GPU script, with reasons:

| override | why |
|---|---|
| `ray_init.num_cpus=32` | **critical** — verl's placement group requests `{CPU: 10, GPU: 1}` per worker bundle (`single_controller/ray/base.py:114`, `max_colocate_count=10`). With auto-detected 8 CPUs the PG is infeasible and `pg.ready()` hangs forever. Oversubscribing logical CPUs is free. |
| `rollout.name=hf` + `+rollout.micro_batch_size=64` | maze recipe's path (`hf_rollout.py` reads `config.get("micro_batch_size")`, sample-level; it divides by n internally -> 4 prompts x 16 = 64 seqs/chunk). No vllm install, no weight-sync, no KV reservation. |
| `val_kwargs.top_k=0` | hf rollout convention (yaml comment: `-1 for vLLM, 0 for HF`). |
| `n=16`, batch 64, resp 1024, 50 steps | fits time/memory budget above. |
| `num_reward_actors=4` | reward manager is built **twice** (train + val, `main_ppo.py:179-180`) so this spawns 8 one-CPU ray actors; 64 as in the 8-GPU script would demand 128 logical CPUs. |
| `+data.dataloader_num_workers=2` | default is 8 per dataloader x 2 dataloaders = 16 procs on 8 cores. |
| `trainer.logger=['console']` | drops the wandb dependency/login; add wandb back if desired. |
| `trainer.total_training_steps=50` | dataset gives 116 steps/epoch at batch 64; explicit cap keeps each cell bounded. |
| no `ray start` | with `nnodes=1` main_ppo passes `address=None` and starts its own instance; external `ray start --num-gpus 8` is both wrong for this box and unused. |

The curriculum plumbing needs no extra flags: `FrontierTeacher` picks up
`n_rollouts=actor_rollout_ref.rollout.n`, and the `observe_batch` feedback path requires
`non_tensor_batch["index"]`, which `rl_dataset.py:322` populates from `extra_info.index` —
exactly what the gsm8k preprocessor writes.

---

## 4. Data prep

`examples/maxrl_data_preprocess/gsm8k.py` exists; args: `--local_dir` (default `~/data/gsm8k`)
and optional `--hdfs_dir`. It loads `openai/gsm8k` (train, 7,473 rows) and
`madrylab/gsm8k-platinum` (test, 1,209 rows) — both confirmed reachable on the hub —
appends the `\boxed{}` instruction, and writes `train.parquet`/`test.parquet` with
`data_source="openai/gsm8k"` (which routes `default_compute_score` to math_verify) and
`extra_info.index` (needed by the curriculum teacher). It imports `verl.utils.hdfs_io`,
i.e. the venv must be installed first (`datasets` is currently NOT importable —
`from datasets import load_dataset` fails on system python).

```bash
source ~/venvs/maxrl311/bin/activate
cd ~/work/curriculumrl/maxrl
python examples/maxrl_data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

# 256-row validation slice (full 1209-prompt platinum x n=4 would cost ~15 min/eval on hf rollout)
python - <<'EOF'
import pandas as pd
df = pd.read_parquet('~/data/gsm8k/test.parquet'.replace('~', __import__('os').path.expanduser('~')))
df.head(256).to_parquet(__import__('os').path.expanduser('~/data/gsm8k/test_256.parquet'))
EOF
```

Model weights: pass the HF id directly (`copy_to_local` forwards non-hdfs strings, and
`hf_tokenizer`/`AutoModel` download to `~/.cache/huggingface`); ~720 MB for 360M.

---

## 5. Risk list

1. **GPU is currently occupied.** A cosmos-framework SFT job (PID 133031) holds **19.8 GB** of
   the A10G right now. Nothing fits until it finishes. (It also means my microbenchmarks of
   effective TFLOPS couldn't run; timing numbers above are analytic estimates.)
2. **Ray placement-group hang on 8 cores** — the CPU:10 bundle issue above. Mitigated by
   `ray_init.num_cpus=32`; if init_workers still stalls >5 min, this is where to look first.
3. **HF-generate stragglers.** A chunk runs until its slowest of 64 sequences stops; a single
   1024-token rambler adds ~20 s/chunk (x16 chunks). If step time blows past ~12 min, drop
   `max_response_length` to 768 or batch to 48 before touching anything else.
4. **First-step OOM in the update phase.** Estimates say ~12 GB peak, but HF generate + FSDP
   summon fragmentation is real; run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   and halve `ppo_micro_batch_size_per_gpu` / `log_prob_micro_batch_size_per_gpu` if hit.
5. **StatefulDataLoader / tensordict versions.** `torchdata==0.11.0` has the stateful loader;
   `tensordict==0.6.2` matches the fork's dockerfile pins. Straying to tensordict >=0.7 risks
   DataProto breakage.
6. **Reward manager double-instantiation** (train+val) doubles ray reward actors; with 4 each
   this is fine, but don't copy the 8-GPU script's 64. Per-item scoring uses SIGALRM inside
   each actor's main thread — safe in single-process ray actors.
7. **Checkpoint disk**: fp32 model+optimizer ~4.4 GB per save for 360M; save_freq=25 +
   keep-2 => ~9 GB/cell, 787 GB free — fine, but don't set save_freq=1.
8. **30 GB RAM**: ray object store (~30% RAM default) + 1,024-seq DataProtos (~100 MB) are
   fine, but avoid running the 2x2 cells concurrently — one run at a time.
9. **`torch.compile` on first step** (`use_torch_compile: true` default, compiles the entropy
   kernel): adds minutes of warmup; if it misbehaves on py3.11/torch2.6, set
   `actor_rollout_ref.actor.use_torch_compile=False`.
10. **MFU logging noise only**: SmolLM2 is `llama` model_type — in FlopsCounter's supported set,
   so no issue; flagged only because other small models would print warnings.

## 6. Go / no-go

**GO**, with one hard precondition and one config landmine:

- Precondition: use the **python3.11 venv** exactly as in §1. The stated python3.9 environment
  is a verified dead end — `math-verify` (a hard import of `verl.workers.reward_manager`)
  crashes at import time on 3.9 with runtime PEP-604 union syntax, in every published version.
  This, not vllm or flash-attn (both have drop-in solutions/wheels), is the **single most
  likely blocker** — anyone who starts `pip3 install --user`-ing into python3.9 will burn
  hours before hitting an unfixable wall.
- Landmine: `ray_init.num_cpus=32`, or the run hangs silently at worker init on this 8-core box.

Everything else checks out: the hf-rollout code path is complete and precedented in the
paper's own maze recipe; memory fits with ~2x headroom for 360M; a 50-step cell lands at
5-8 h; data and model downloads were verified reachable; and the curriculum sampler's
index/uid feedback loop is wired end-to-end for the gsm8k parquet schema.
