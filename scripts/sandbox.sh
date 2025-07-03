#!/bin/bash

# export VLLM_ATTENTION_BACKEND=XFORMERS
ulimit -c 0

# Wait until gpu 0 memory < 1000
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    if [ "$MEM" -lt 10 ]; then
        break
    fi
    echo "Waiting for GPU memory to be less than 10 MB, current usage: $MEM MB"
    sleep 10
done

set -x

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# export NCCL_P2P_DISABLE=1 # https://github.com/volcengine/verl/issues/597#issuecomment-2731472061

export ALL_PROXY=http://a800-1:7890 HTTPS_PROXY=http://a800-1:7890 NO_PROXY="10.0.1.5"
wandb login "7f15726daf2e9c716a1b32198b5a0de9dbfbcdf5"

export RAY_TMPDIR=/mnt/data1/zhengxin2020/tmp_ray_zx
export RAY_local_fs_capacity_threshold=1 

MODEL_DIR="/mnt/data1/hf_models"
MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B" # "deepseek-llm-7b-chat" # "Qwen2.5-3B-Instruct"
DATA_NAME="verifiable-coding-problems-python-only" # "verifiable-coding-problems"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.sandbox_fusion.url='http://10.0.1.5:8081/common_evaluate_batch' \
    reward_model.sandbox_fusion.max_concurrent=64 \
    reward_model.reward_manager=prime \
    data.train_files=data/${DATA_NAME}/train.parquet \
    data.val_files=data/${DATA_NAME}/validation.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_DIR}/${MODEL_NAME} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_example_sandbox_fusion' \
    trainer.experiment_name="${DATA_NAME}_${MODEL_NAME}_function_sandbox_fusion_grpo" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    trainer.total_epochs=15 $@
