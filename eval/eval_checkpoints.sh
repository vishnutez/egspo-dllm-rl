#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=4
#SBATCH --output=logs_eval_checkpoints/%x_%j.out

# Optionally load modules, e.g.: Miniconda3 and CUDA

# Activate conda environment
source activate <YOUR_CONDA_ENV_NAME \>

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}  # Set CUDA_HOME if not already set


export HF_HOME=<YOUR_HF_HOME_DIR>  # remove this line if you are using the default HF_HOME
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


MASTER_PORT=29411

# Arrays of tasks and generation lengths
TASKS=("sudoku")
GEN_LENGTHS=(256 512 128)

# Multi-node settings derived from SLURM
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_PROCID: $SLURM_PROCID"

NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "SLURM multi-node: nnodes=$NNODES master_addr=$MASTER_ADDR master_port=$MASTER_PORT"

# GPUs per node (from Slurm allocation)
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
echo "Using $NUM_GPUS GPUs per node"

CHECKPOINT_DIR="<YOUR_CHECKPOINT_DIR>"
NUM_CHECKPOINTS=20  # Evaluation will be done on the last NUM_CHECKPOINTS checkpoints
OUTPUT_DIR="checkpoints/<YOUR_TASK>/<YOUR_CHECKPOINT_DIR>"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    diffusion_steps=$((gen_length / 2))
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size, diffusion_steps=$diffusion_steps across $NNODES nodes"
    srun --ntasks=$NNODES --ntasks-per-node=2 --kill-on-bad-exit=1 \
      bash -lc "torchrun \
        --nnodes $NNODES \
        --nproc_per_node $NUM_GPUS \
        --node_rank \$SLURM_PROCID \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        eval_checkpoints.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --diffusion_steps $diffusion_steps \
        --output_dir $OUTPUT_DIR \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --checkpoint_dir $CHECKPOINT_DIR \
        --num_checkpoints $NUM_CHECKPOINTS
      "
    done
done

echo "All evaluations completed!"
