#!/bin/bash
# GSM8K 2x2 on a single A10G — from curriculum_maxrl/GSM8K_A10G_PLAN.md.
# Usage: ESTIMATOR=maxrl CURRICULUM=true bash smollm/smollm_a10g.sh
# Requires: source ~/venvs/maxrl311/bin/activate  (py311 venv, see plan §1)
set -e
source ~/venvs/maxrl311/bin/activate
export HF_HOME=~/hf-cache

MODEL_PATH=HuggingFaceTB/SmolLM2-360M-Instruct
TRAIN_DATA=$HOME/data/gsm8k/train.parquet
VAL_DATA=$HOME/data/gsm8k/test_256.parquet
ADVANTAGE_ESTIMATOR=${ESTIMATOR:-maxrl}          # maxrl | grpo
CURRICULUM=${CURRICULUM:-true}                   # true | false
N_ROLLOUTS=16
STEPS=${STEPS:-50}

cd "$(dirname "$0")/.."

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
  +data.curriculum.decay=0.7 \
  +data.curriculum.success_threshold=0.5 \
  +data.curriculum.utility=advmass \
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
  trainer.total_training_steps=${STEPS} \
  trainer.save_freq=25 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=25 \
  trainer.default_local_dir=$HOME/ckpt/gsm8k_a10g/${ADVANTAGE_ESTIMATOR}_cur${CURRICULUM}
