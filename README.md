# EGSPO-SA

Official implementation of Entropy-Guided Stepwise Policy Optimization with Stepwise Advantages (EGSPO-SA) for RL fine-tuning of diffusion Large Language Models (dLLMs). In this repo, we call it EPSA for short.

## Setup

1. Clone the repository
2. Install dependencies: `conda env create -f environment.yml`

## Training

Configure environment variables (wandb, hf_home, etc.) in `epsa/train.sh`. We provide sbatch file, which can be converted to .sh file easily

```bash
sbatch epsa/train.sh
```

Default configurations are used in experiments.

## Evaluation

1. Generate completions: Update checkpoint directory in `eval/eval_checkpoints.sh` and run
2. Compute metrics: Modify `task, checkpoint_dir, generated_lengths` in `eval/get_and_save_metrics.py` using those evaluated in the last step and run it to get saved metrics in `.json` files
