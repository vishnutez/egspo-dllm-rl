#!/bin/bash
#SBATCH --job-name=<YOUR_JOB_NAME>
#SBATCH --time=96:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=128G # 128GB
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=4
#SBATCH --output=logs/%x_%j.out

# ----------------------------
# User-configurable parameters
# ----------------------------
MASTER_PORT=$((12000 + RANDOM % 1000))
CFGDIR="./accel_cfg"
PRECISION="bf16"                  # bf16 | fp16 | no

# DeepSpeed config options
DS_ZERO_STAGE=2
DS_OVERLAP_COMM=true
DS_GRAD_CLIP=1
DS_OFFLOAD_OPT="none"
DS_OFFLOAD_PARAM="none"
DS_ZERO3_INIT=false

DATASET="gsm8k"

MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct

NUM_GENERATIONS=8
PER_DEVICE_TRAIN_BATCH_SIZE=6
GRAD_ACCUMULATION_STEPS=2
LOGPS_EVAL_NUM_STEPS=8
LEARNING_RATE=3e-5
LOGPS_EVAL_MODE="unbiased"
LOGPS_EVAL_TIME_STEPS_MODE="high_entropy"
EPSILON=0.5
TEMPERATURE=0.9
TERMINATE_AT_LAST_NON_EOS=False
BETA=0.04
LAMBDA1=0.5
NORMALIZE_RETURNS=False
USE_EXACT_KL=False
CORRECTNESS_STEP_REWARD_ONLY=False
LOGPS_AGGREGATION_MODE="mean"

if [ "$TERMINATE_AT_LAST_NON_EOS" = True ]; then
    TERMINATE_AT_LAST_NON_EOS_FLAG="_terminate_eos"
else
    TERMINATE_AT_LAST_NON_EOS_FLAG=""
fi

if [ "$USE_EXACT_KL" = True ]; then
    KL_TYPE="exact_kl"
else
    KL_TYPE="kl"
fi

if (( $(echo "$LAMBDA1 > 0.0" | bc -l) )); then
    ALGO_NAME="epsa_lambda1_${LAMBDA1}_"
elif [ "$LOGPS_EVAL_TIME_STEPS_MODE" == "high_entropy" ]; then
    ALGO_NAME="ep_lambda1_0.0_"
else
    ALGO_NAME="vanilla_grpo_"
fi

if [ "$CORRECTNESS_STEP_REWARD_ONLY" = True ]; then
    STEP_REWARD_ONLY_FLAG="_correctness_step_reward_only"
else
    STEP_REWARD_ONLY_FLAG=""
fi

RUN_NAME="${ALGO_NAME}${LOGPS_EVAL_MODE}_${LOGPS_EVAL_TIME_STEPS_MODE}_${DATASET}_eps_${EPSILON}_temp_${TEMPERATURE}_ng${NUM_GENERATIONS}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRAD_ACCUMULATION_STEPS}_le${LOGPS_EVAL_NUM_STEPS}_lr${LEARNING_RATE}_${KL_TYPE}_${BETA}${STEP_REWARD_ONLY_FLAG}_logps_aggregation_${LOGPS_AGGREGATION_MODE}"

# epsa_train.py args (editable)
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
                  --temperature ${TEMPERATURE} \
                  --terminate_at_last_non_eos ${TERMINATE_AT_LAST_NON_EOS} \
                  --beta ${BETA} \
                  --use_exact_kl ${USE_EXACT_KL} \
                  --lambda1=${LAMBDA1} \
                  --normalize_returns ${NORMALIZE_RETURNS} \
                  --correctness_step_reward_only ${CORRECTNESS_STEP_REWARD_ONLY} \
                  --logps_aggregation_mode ${LOGPS_AGGREGATION_MODE}"
                  

# ----------------------------
# Environment setup
# ----------------------------

# Optinally load modules, e.g.: Miniconda3 and CUDA

source activate <YOUR_CONDA_ENV_NAME \>

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export WANDB_PROJECT=<YOUR_WANDB_PROJECT>
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}  # Set CUDA_HOME if not already set
export HF_HOME=<YOUR_HF_HOME_DIR>  # remove this line if you are using the default HF_HOME

mkdir -p logs "$CFGDIR"

# ----------------------------
# Cluster topology
# ----------------------------
NUM_MACHINES=${SLURM_NNODES:?SLURM_NNODES not set}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
TOTAL_PROCS=$((NUM_MACHINES * GPUS_PER_NODE))

echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"
echo "NUM_MACHINES=$NUM_MACHINES  GPUS_PER_NODE=$GPUS_PER_NODE  TOTAL_PROCS=$TOTAL_PROCS"

# ----------------------------
# Generate per-machine Accelerate configs
# ----------------------------
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

# ----------------------------
# Launch with srun (one task per node)
# ----------------------------
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"

echo "Launching training with srun (Accelerate + DeepSpeed)..."
srun \
  --ntasks="$NUM_MACHINES" \
  --nodes="$NUM_MACHINES" \
  --ntasks-per-node=1 \
  bash -lc '
    export WANDB_API_KEY='"$WANDB_API_KEY"'
    export WANDB_PROJECT='"$WANDB_PROJECT"'
    export CUDA_HOME='"$CUDA_HOME"'
    export HF_HOME='"$HF_HOME"'
    export HF_DATASETS_OFFLINE='"$HF_DATASETS_OFFLINE"'
    export TRANSFORMERS_OFFLINE='"$TRANSFORMERS_OFFLINE"'
    ID=${SLURM_PROCID}
    echo ">>> Node $(hostname) starting machine_rank=${ID}"
    accelerate launch \
      --config_file '"$CFGDIR"'/accelerate_machine_${ID}.yaml \
      --main_process_port '"$MASTER_PORT"' \
      '"$TRAIN_SCRIPT"' \
      '"$TRAIN_ARGS_BASE"' \
      '"$TRAIN_ARGS_EXTRA"'
  '

echo "All nodes completed."
