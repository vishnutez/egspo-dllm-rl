#!/bin/bash
#SBATCH --job-name=egsposa_train
#SBATCH --time=96:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out

# ============================================
# User-configurable parameters
# ============================================
MASTER_PORT=$((12000 + RANDOM % 1000))
CFGDIR="./accel_cfg"
PRECISION="bf16"   # bf16 | fp16 | no

# DeepSpeed config
DS_ZERO_STAGE=2
DS_OVERLAP_COMM=true
DS_GRAD_CLIP=1
DS_OFFLOAD_OPT="none"
DS_OFFLOAD_PARAM="none"
DS_ZERO3_INIT=false

# Experiment config
DATASET="gsm8k"
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"

NUM_GENERATIONS=8
PER_DEVICE_TRAIN_BATCH_SIZE=6
GRAD_ACCUMULATION_STEPS=2
LOGPS_EVAL_NUM_STEPS=8
LEARNING_RATE=3e-5
LOGPS_EVAL_MODE="unbiased"
LOGPS_EVAL_TIME_STEPS_MODE="high_entropy"
EPSILON=0.2
EPSILON_HIGH=0.5
TEMPERATURE=0.9
LAMBDA1=0.5
NORMALIZE_RETURNS=true

# --------------------------------------------
# Optional environment variables
# Set these before launching if needed:
#   export WANDB_API_KEY=...
#   export WANDB_PROJECT=...
#   export HF_HOME=...
# --------------------------------------------
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-egsposa}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# ============================================
# Derived names
# ============================================
if (( $(echo "$LAMBDA1 > 0.0" | bc -l) )); then
    if [ "$NORMALIZE_RETURNS" = true ]; then
        NORMALIZE_RETURNS_FLAG="_normalized"
    else
        NORMALIZE_RETURNS_FLAG=""
    fi
    ALGO_NAME="epsa_lambda1_${LAMBDA1}${NORMALIZE_RETURNS_FLAG}"
elif [ "$LOGPS_EVAL_TIME_STEPS_MODE" = "high_entropy" ]; then
    ALGO_NAME="ep_lambda1_0.0"
else
    ALGO_NAME="vanilla_grpo"
fi

RUN_NAME="${ALGO_NAME}_${LOGPS_EVAL_MODE}_${LOGPS_EVAL_TIME_STEPS_MODE}_${DATASET}_eps_${EPSILON}_epshigh_${EPSILON_HIGH}_temp_${TEMPERATURE}_ng${NUM_GENERATIONS}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRAD_ACCUMULATION_STEPS}_le${LOGPS_EVAL_NUM_STEPS}_lr${LEARNING_RATE}"

# ============================================
# Train script arguments
# Some options are expected to be defined in
# slurm_scripts/train.yaml instead of here.
# ============================================
TRAIN_SCRIPT="epsa_train.py"
TRAIN_ARGS_BASE="--config slurm_scripts/train.yaml"
TRAIN_ARGS_EXTRA="--model_path ${MODEL_PATH} \
                  --dataset ${DATASET} \
                  --run_name ${RUN_NAME} \
                  --output_dir checkpoints/${DATASET}/${RUN_NAME} \
                  --num_generations ${NUM_GENERATIONS} \
                  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                  --generation_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
                  --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
                  --logps_eval_num_steps ${LOGPS_EVAL_NUM_STEPS} \
                  --logps_eval_mode ${LOGPS_EVAL_MODE} \
                  --logps_eval_time_steps_mode ${LOGPS_EVAL_TIME_STEPS_MODE} \
                  --learning_rate ${LEARNING_RATE} \
                  --epsilon ${EPSILON} \
                  --epsilon_high ${EPSILON_HIGH} \
                  --temperature ${TEMPERATURE} \
                  --lambda1 ${LAMBDA1} \
                  --normalize_returns ${NORMALIZE_RETURNS}"

# ============================================
# Environment setup
# Uncomment/adapt these lines if your cluster
# requires modules.
# ============================================
# module load Miniconda3
# module load CUDA/12.9.0

mkdir -p logs
mkdir -p "${CFGDIR}"

# Activate conda environment
# Replace "your_env_name" with your actual environment name.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate your_env_name
else
    echo "ERROR: conda not found. Please install Conda or load it on your cluster."
    exit 1
fi

# ============================================
# Cluster topology
# ============================================
NUM_MACHINES="${SLURM_NNODES:?SLURM_NNODES not set}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
TOTAL_PROCS=$((NUM_MACHINES * GPUS_PER_NODE))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NUM_MACHINES=$NUM_MACHINES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "TOTAL_PROCS=$TOTAL_PROCS"

# ============================================
# Generate per-machine Accelerate configs
# ============================================
for RANK in $(seq 0 $((NUM_MACHINES - 1))); do
cat > "${CFGDIR}/accelerate_machine_${RANK}.yaml" <<EOF
compute_environment: LOCAL_MACHINE
debug: false

distributed_type: DEEPSPEED
main_training_function: main
mixed_precision: '${PRECISION}'
downcast_bf16: 'auto'

rdzv_backend: static
same_network: true
main_process_ip: ${MASTER_ADDR}
main_process_port: ${MASTER_PORT}

num_machines: ${NUM_MACHINES}
num_processes: ${TOTAL_PROCS}
machine_rank: ${RANK}
gpu_ids: all
use_cpu: false

deepspeed_config:
  deepspeed_multinode_launcher: standard
  zero_stage: ${DS_ZERO_STAGE}
  zero3_init_flag: ${DS_ZERO3_INIT}
  offload_optimizer_device: ${DS_OFFLOAD_OPT}
  offload_param_device: ${DS_OFFLOAD_PARAM}
  overlap_comm: ${DS_OVERLAP_COMM}
  gradient_clip: ${DS_GRAD_CLIP}

tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
EOF
done

# ============================================
# Launch with srun
# ============================================
export MASTER_ADDR
export MASTER_PORT

echo "Launching training with srun (Accelerate + DeepSpeed)..."

srun \
  --ntasks="${NUM_MACHINES}" \
  --nodes="${NUM_MACHINES}" \
  --ntasks-per-node=1 \
  bash -lc '
    export WANDB_API_KEY='"${WANDB_API_KEY}"'
    export WANDB_PROJECT='"${WANDB_PROJECT}"'
    export HF_HOME='"${HF_HOME}"'
    export CUDA_HOME='"${CUDA_HOME}"'

    ID=${SLURM_PROCID}
    echo ">>> Node $(hostname) starting machine_rank=${ID}"

    accelerate launch \
      --config_file '"${CFGDIR}"'/accelerate_machine_${ID}.yaml \
      --main_process_port '"${MASTER_PORT}"' \
      '"${TRAIN_SCRIPT}"' \
      '"${TRAIN_ARGS_BASE}"' \
      '"${TRAIN_ARGS_EXTRA}"'
  '

echo "All nodes completed."
