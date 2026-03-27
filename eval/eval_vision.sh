#!/bin/bash
#SBATCH --job-name=egsa_eval
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H200:8
#SBATCH --mem=256G
#SBATCH --output=logs_eval_checkpoints/%x_%j.out

# ============================================
# User-configurable parameters
# ============================================

RUN_NAME="egspo_lambda1_0.1_correctness_step_reward_only_unbiased_high_entropy_gsm8k_eps_0.5_epshigh_0.5_temp_0.9_ng8_bs6_ga8_le8_lr3e-5_kl0.04_logps_aggregation_mean"
CHECKPOINT_DIR="../egspo/checkpoints/gsm8k/${RUN_NAME}"
NUM_CHECKPOINTS=12  # Evaluation will be done on the last NUM_CHECKPOINTS checkpoints
OUTPUT_DIR="checkpoints/gsm8k/${RUN_NAME}"

TASKS=("gsm8k")
GEN_LENGTHS=(256 512 128)

MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"

# --------------------------------------------
# Optional environment variables
#   export HF_HOME=...
# --------------------------------------------

export HF_HOME="${HF_HOME:-/scratch/user/vishnukunde_tamu.edu/.cache/huggingface}"
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"

# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# ============================================
# Environment setup
# ============================================

module load Miniforge3/25.3.0-3
module load CUDA/12.9.1

mkdir -p logs_eval_checkpoints

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/user/vishnukunde_tamu.edu/conda/envs/egspo-env

# ============================================
# Launch — single node, direct torchrun
# ============================================

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
NUM_GPUS=${NUM_GPUS:-8}
echo "Launching single-node evaluation on $(hostname) with $NUM_GPUS GPUs..."

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    diffusion_steps=$((gen_length / 2))
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size, diffusion_steps=$diffusion_steps"
    torchrun \
      --nproc_per_node $NUM_GPUS \
      eval_checkpoints.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --diffusion_steps $diffusion_steps \
      --output_dir $OUTPUT_DIR \
      --model_path $MODEL_PATH \
      --checkpoint_dir $CHECKPOINT_DIR \
      --num_checkpoints $NUM_CHECKPOINTS
  done
done

echo "All evaluations completed!"
