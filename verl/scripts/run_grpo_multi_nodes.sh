#!/bin/bash

#SBATCH --job-name=LLM-RL-Training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=30-00:00:00
#SBATCH --cpus-per-gpu=12
#SBATCH --gpus-per-node=8
#SBATCH --mem=200G
#SBATCH --output=logs/train.out

set -x

unset VLLM_ATTENTION_BACKEND

# === User Configuration ===
CONTAINER_NAME="llm_training_container"
IMG="rocm6.3.4:vllm-0.8.5-numa-patch-ubuntu-22.04-cascade-fix"
TIME=$(date +%Y%m%d_%H%M%S)

# === Launch Docker Container ===
srun bash -c "
    docker image prune -f

    if ! docker images --format \"{{.Repository}}:{{.Tag}}\" | grep -q \"${IMG}\"; then
        echo \"Loading Docker image ${IMG}...\"
        docker load -i /path/to/your_image.tar
    else
        echo \"Docker image ${IMG} already exists.\"
    fi

    ibdev2netdev

    docker run --rm -d \
        --env-file scripts/multi_node_env.list \
        --network host \
        --device /dev/dri \
        --device /dev/kfd \
        --device /dev/infiniband \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --privileged \
        -v /data:/data \
        -v /mnt/blob:/mnt/blob \
        -v \${HOME}/.ssh:/root/.ssh \
        -w /data \
        --shm-size 128G \
        --name \"${CONTAINER_NAME}\" \
        \"${IMG}\" \
        tail -f /dev/null

    echo \"Container launched.\"
"

# === Setup Ray Cluster ===
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    head_node_ip=${#ADDR[0]} -gt 16 ? ${ADDR[1]} : ${ADDR[0]}
fi

ip_head=$head_node_ip:7777
export ip_head

printenv

srun --nodes=1 --ntasks=1 -w "$head_node" docker exec "${CONTAINER_NAME}" ray stop
srun --nodes=1 --ntasks=1 -w "$head_node" docker exec "${CONTAINER_NAME}" ray start \
    --head --node-ip-address="$head_node_ip" --port=7777 \
    --dashboard-host=0.0.0.0 --dashboard-port=5555 \
    --num-cpus "${SLURM_CPUS_PER_GPU}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &

sleep 60

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w "$node_i" docker exec "${CONTAINER_NAME}" ray start \
        --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_GPU}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done

# === Test Ray Cluster ===
docker exec "${CONTAINER_NAME}" python3 -c '
import ray
try:
    ray.init(address="auto")
    print("\n=== Ray Cluster Status ===")
    for node in ray.nodes():
        print(f"Node: {node[\"NodeManagerHostname\"]}, Status: {node[\"Alive\"]}")
    ray.shutdown()
except Exception as e:
    print(f"Ray init failed: {str(e)}")
'

# === Configure Training ===
PROJECT_NAME=llm-project
RUN_NAME=rl-training-${SLURM_NNODES}node
DATA_DIR=/data/datasets
MODEL_PATH=/mnt/blob/checkpoints/initial_model
OUTPUT_PATH=/mnt/blob/results/${PROJECT_NAME}/${RUN_NAME}

train_files="['$DATA_DIR/train.parquet']"
test_files="['$DATA_DIR/test.parquet']"

PYTHONUNBUFFERED=1 srun --overlap --nodes=${SLURM_NNODES} --ntasks=1 -w "$head_node" \
    docker exec -w /data/code "${CONTAINER_NAME}" \
    python3 -m your_module.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=1312 \
    data.max_prompt_length=512 \
    data.max_response_length=16000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=17000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.attn_config.use_cascade=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=17000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    reward_model.enable=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.default_local_dir=$OUTPUT_PATH \
    trainer.val_before_train=True \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 \
    trainer.validation_data_dir=$OUTPUT_PATH/validation
