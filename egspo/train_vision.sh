#!/bin/bash
#SBATCH --job-name=ada_egspo_train
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:8
#SBATCH --mem=256G
#SBATCH --output=logs/%x_%j.out

# ============================================
# User-configurable parameters
# ============================================

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
ALPHA_ENTROPY=0.7

# --------------------------------------------
# Optional environment variables
#   export WANDB_API_KEY=...
#   export WANDB_PROJECT=...
#   export HF_HOME=...
# --------------------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"
export HF_HOME="${HF_HOME:-/scratch/user/vishnukunde_tamu.edu/.cache/huggingface}"
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"
# export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# ============================================
# Derived run name
# ============================================
if (( $(echo "$LAMBDA1 > 0.0" | bc -l) )); then
    if [ "$NORMALIZE_RETURNS" = true ]; then
        NORMALIZE_RETURNS_FLAG="_normalized"
    else
        NORMALIZE_RETURNS_FLAG=""
    fi
    ALGO_NAME="ada_egspo_lambda1_${LAMBDA1}${NORMALIZE_RETURNS_FLAG}_alpha${ALPHA_ENTROPY}"
elif [ "$LOGPS_EVAL_TIME_STEPS_MODE" = "high_entropy" ]; then
    ALGO_NAME="ada_ep_lambda1_0.0_alpha${ALPHA_ENTROPY}"
else
    ALGO_NAME="ada_vanilla_grpo_alpha${ALPHA_ENTROPY}"
fi

RUN_NAME="${ALGO_NAME}_${LOGPS_EVAL_MODE}_${LOGPS_EVAL_TIME_STEPS_MODE}_${DATASET}_eps_${EPSILON}_epshigh_${EPSILON_HIGH}_temp_${TEMPERATURE}_ng${NUM_GENERATIONS}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRAD_ACCUMULATION_STEPS}_le${LOGPS_EVAL_NUM_STEPS}_lr${LEARNING_RATE}"

# ============================================
# Environment setup
# ============================================

module load Miniforge3/25.3.0-3
module load CUDA/12.9.1

mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/user/vishnukunde_tamu.edu/conda/envs/egspo-env

# ============================================
# Launch — single node, direct accelerate launch
# ============================================
echo "Launching single-node training on $(hostname)..."
echo "RUN_NAME=${RUN_NAME}"

accelerate launch \
    --config_file slurm_scripts/accelerate_a100.yaml \
    ada_egspo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path ${MODEL_PATH} \
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
    --normalize_returns ${NORMALIZE_RETURNS} \
    --alpha_entropy ${ALPHA_ENTROPY}
