#!/bin/bash
# Multi-node training script for 16*H100 (2 nodes x 8 GPUs)
# Successfully tested with InfiniBand/RDMA communication
set -x

# ==============================================================================
# CLUSTER CONFIG
# ==============================================================================
HEAD_NODE_IP="172.27.21.45"
WORKER_NODE_IP="172.27.25.162"
RAY_PORT=6379
NNODES=2
GPUS_PER_NODE=8

# ==============================================================================
# TRAINING CONFIG
# ==============================================================================
MAX_EPOCHS=8
DATASET=code-r1-12k
MODEL_PATH=/home/ubuntu/code-r1/models/Qwen2.5-7B-Instruct-1M
ROLLOUT_N_SAMPLE=16
ROLLOUT_N_QUERY=16
MICRO_BATCH_PER_GPU=8
GRAD_ACC_STEPS=2

TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
GLOBAL_BATCH_SIZE=$((TOTAL_GPUS * MICRO_BATCH_PER_GPU * GRAD_ACC_STEPS))

# Validate batch size
TOTAL_SAMPLES=$((ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE))
if (( TOTAL_SAMPLES % GLOBAL_BATCH_SIZE != 0 )); then
    echo "Error: (ROLLOUT_N_QUERY * ROLLOUT_N_SAMPLE) must be divisible by GLOBAL_BATCH_SIZE."
    exit 1
fi

# ==============================================================================
# ENVIRONMENT
# ==============================================================================
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY="7b086d88f5a4dada0caedfc027e3eb69f166b941"
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

# ==============================================================================
# SETUP RAY CLUSTER
# ==============================================================================
setup_ray_cluster() {
    echo "Setting up Ray cluster..."
    
    # Stop any existing Ray processes
    ssh ${WORKER_NODE_IP} "ray stop --force" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    sleep 2
    
    # Start Ray head on this node
    echo "Starting Ray head on ${HEAD_NODE_IP}..."
    ray start --head --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE}
    sleep 3
    
    # Start Ray worker on the second node
    echo "Starting Ray worker on ${WORKER_NODE_IP}..."
    ssh ${WORKER_NODE_IP} "ray start --address='${HEAD_NODE_IP}:${RAY_PORT}' --num-gpus=${GPUS_PER_NODE}"
    sleep 3
    
    # Verify cluster
    ray status
}

nccl_test() {
    echo "Running NCCL test to verify InfiniBand connectivity..."
    /opt/hpcx/ompi/bin/mpirun --prefix /opt/hpcx/ompi --hostfile ~/hostfile.txt -np 16 \
        -x LD_LIBRARY_PATH -mca pml ucx \
        ~/nccl-tests/build/all_reduce_perf -b 1M -e 1M -f 2 -g 1 -c 1 -n 1
}

# ==============================================================================
# MAIN
# ==============================================================================
case "${1:-run}" in
    setup)
        setup_ray_cluster
        ;;
    stop)
        ray stop --force
        ssh ${WORKER_NODE_IP} "ray stop --force"
        ;;
    test)
        nccl_test
        ;;
    run)
        # Multi-node training (16 GPUs)
        if ! ray status &>/dev/null; then
            setup_ray_cluster
        fi
        
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=/home/ubuntu/code-r1/data/$DATASET/train.parquet \
            data.val_files=/home/ubuntu/code-r1/data/$DATASET/test.parquet \
            data.train_batch_size=$ROLLOUT_N_QUERY \
            data.max_prompt_length=2048 \
            data.max_response_length=4096 \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.actor.optim.lr=5e-7 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=$GLOBAL_BATCH_SIZE \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.log_prob_micro_batch_size=256 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
            actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
            actor_rollout_ref.ref.log_prob_micro_batch_size=256 \
            actor_rollout_ref.ref.fsdp_config.param_offload=False \
            algorithm.kl_ctrl.kl_coef=0.001 \
            trainer.critic_warmup=0 \
            trainer.logger=['wandb'] \
            trainer.project_name='code-r1' \
            trainer.experiment_name=${DATASET}-grpo-16h100 \
            trainer.nnodes=$NNODES \
            trainer.default_local_dir=/home/ubuntu/code-r1/models/${DATASET}-grpo \
            trainer.n_gpus_per_node=$GPUS_PER_NODE \
            trainer.save_freq=64 \
            trainer.test_freq=16 \
            trainer.total_epochs=$MAX_EPOCHS \
            reward_model.reward_manager=prime ${@:2} 2>&1 | tee grpo_multinode.log
        ;;
    *)
        echo "Usage: $0 {setup|stop|test|run}"
        echo "  setup - Setup Ray cluster on both nodes"
        echo "  stop  - Stop Ray cluster"
        echo "  test  - Run NCCL performance test"
        echo "  run   - Run training on multi-node (16 GPUs)"
        exit 1
        ;;
esac
