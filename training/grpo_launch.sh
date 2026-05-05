#!/usr/bin/env bash
# GRPO training launcher for ToolRL-DR-Full / ToolRL-DR-Mixed.
#
# Reproduces the ToolRL public GRPO recipe with our perturbation-augmented
# training data substituted in. Assumes the public ToolRL repository has
# been cloned and is on PYTHONPATH so that `verl.trainer.main_ppo` is
# importable; see training/README.md for setup.
#
# Required env vars:
#   TRAIN_FILE      path to one of training/data/{train,mixed_train}.parquet
#                   (train.parquet -> ToolRL-DR-Full; mixed_train.parquet ->
#                   ToolRL-DR-Mixed)
#   VAL_FILE        path to training/data/test.parquet
#   BASE_MODEL      path or HF id of the backbone (Qwen/Qwen2.5-3B-Instruct)
#   EXPERIMENT_NAME human-readable run id, e.g. dr_full or dr_mixed
#   N_GPUS          number of GPUs visible to this job (we use 2)
#   ROLLOUT_TP_SIZE tensor-parallel size for vLLM rollout (we use 1)
#
# Example: ToolRL-DR-Full
#   TRAIN_FILE=training/data/train.parquet \
#   VAL_FILE=training/data/test.parquet \
#   BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
#   EXPERIMENT_NAME=toolrl-dr-full \
#   N_GPUS=2 ROLLOUT_TP_SIZE=1 \
#   bash training/grpo_launch.sh
#
# Example: ToolRL-DR-Mixed
#   TRAIN_FILE=training/data/mixed_train.parquet ... bash training/grpo_launch.sh

set -euo pipefail

: "${TRAIN_FILE:=training/data/train.parquet}"
: "${VAL_FILE:=training/data/test.parquet}"
: "${BASE_MODEL:?set BASE_MODEL to the backbone model id or path}"
: "${EXPERIMENT_NAME:?set EXPERIMENT_NAME for the run}"
: "${N_GPUS:=2}"
: "${ROLLOUT_TP_SIZE:=1}"

export VLLM_ATTENTION_BACKEND=XFORMERS

# All reward-shaping switches are 0 (we use the base ToolRL reward).
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

# Hyperparameters match the ToolRL public recipe exactly. Only the
# data.train_files path differs between DR-Full and DR-Mixed.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=512 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=5 \
    trainer.total_epochs=15
