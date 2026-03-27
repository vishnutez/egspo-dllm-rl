#!/bin/bash
#SBATCH --job-name=egsposa_train
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
LEARNING_RATE=3e-5
LOGPS_EVAL="unbiased"
LOGPS_EVAL_STEP_SELECTION="high_entropy"
EPSILON=0.5
EPSILON_HIGH=0.5
TEMPERATURE=0.9
LAMBDA1=0.1
NORMALIZE_RETURNS=false
GRAD_ACCUMULATION_STEPS=8
LOGPS_EVAL_NUM_STEPS=8
BETA=0.04
LOGPS_AGGREGATION="mean"
CORRECTNESS_STEP_REWARD_ONLY=true
STEPWISE_LAMBDA1=false

# --------------------------------------------
# Optional environment variables
#   export WANDB_API_KEY=...
#   export WANDB_PROJECT=...
#   export HF_HOME=...
# --------------------------------------------

export WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"
export WANDB_RESUME="allow"
export WANDB_ID="bo0dp1yq"
export WANDB_MODE="online"
export WANDB_DIR="${WANDB_DIR:-/scratch/user/vishnukunde_tamu.edu/codebase/egspo-dllm-rl/egspo/wandb}"
mkdir -p "${WANDB_DIR}"
export HF_HOME="${HF_HOME:-/scratch/user/vishnukunde_tamu.edu/.cache/huggingface}"
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "WANDB_DIR=${WANDB_DIR}"
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
    if [ "$CORRECTNESS_STEP_REWARD_ONLY" = true ]; then
        CORRECTNESS_STEP_REWARD_ONLY_FLAG="_correctness_step_reward_only"
    else
        CORRECTNESS_STEP_REWARD_ONLY_FLAG=""
    fi
    if [ "$STEPWISE_LAMBDA1" = true ]; then
        STEPWISE_LAMBDA1_FLAG="_stepwise_lambda1"
    else
        STEPWISE_LAMBDA1_FLAG=""
    fi
    ALGO_NAME="egspo_lambda1_${LAMBDA1}${NORMALIZE_RETURNS_FLAG}${CORRECTNESS_STEP_REWARD_ONLY_FLAG}${STEPWISE_LAMBDA1_FLAG}"
elif [ "$LOGPS_EVAL_STEP_SELECTION" = "high_entropy" ]; then
    ALGO_NAME="ep_lambda1_0.0"
else
    ALGO_NAME="vanilla_grpo"
fi

RUN_NAME="${ALGO_NAME}_${LOGPS_EVAL}_${LOGPS_EVAL_STEP_SELECTION}_${DATASET}_eps_${EPSILON}_epshigh_${EPSILON_HIGH}_temp_${TEMPERATURE}_ng${NUM_GENERATIONS}_bs${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRAD_ACCUMULATION_STEPS}_le${LOGPS_EVAL_NUM_STEPS}_lr${LEARNING_RATE}_kl${BETA}_logps_aggregation_${LOGPS_AGGREGATION}"

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
    egspo_train.py \
    --config slurm_scripts/train.yaml \
    --model_path ${MODEL_PATH} \
    --dataset ${DATASET} \
    --run_name ${RUN_NAME} \
    --output_dir checkpoints/${DATASET}/${RUN_NAME} \
    --num_generations ${NUM_GENERATIONS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --generation_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --logps_eval ${LOGPS_EVAL} \
    --logps_eval_step_selection ${LOGPS_EVAL_STEP_SELECTION} \
    --learning_rate ${LEARNING_RATE} \
    --epsilon ${EPSILON} \
    --epsilon_high ${EPSILON_HIGH} \
    --temperature ${TEMPERATURE} \
    --lambda1 ${LAMBDA1} \
    --normalize_returns ${NORMALIZE_RETURNS} \
    --correctness_step_reward_only ${CORRECTNESS_STEP_REWARD_ONLY} \
    --logps_aggregation ${LOGPS_AGGREGATION} \
    --beta ${BETA} \
    --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
    --logps_eval_num_steps ${LOGPS_EVAL_NUM_STEPS} \
    --stepwise_lambda1 ${STEPWISE_LAMBDA1}