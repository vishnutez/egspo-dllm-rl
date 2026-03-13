## 🚀 Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages

**Entropy-Guided Stepwise Policy Optimization with Stepwise Advantages (EGSPO-SA)** introduces a reinforcement learning framework for **diffusion language models (dLLMs)** 🤖. Unlike autoregressive LLMs, diffusion models generate sequences through an iterative denoising process, making standard sequence-level RL fine-tuning challenging.

We formulate the denoising trajectory as a **finite-horizon Markov decision process** 🔁 and derive a policy-gradient objective that decomposes across denoising steps. Our method focuses learning on the most informative steps and introduces a lightweight **stepwise advantage estimator** ⚡ for efficient training.


## ✨ Key Contributions

- 🧠 **Diffusion-MDP formulation** for RL fine-tuning of diffusion language models  
- 📊 **Entropy-guided step selection** to identify the most informative denoising steps  
- ⚡ **EGSPO-SA**, a lightweight stepwise advantage estimator that avoids separate value models  
- 🏆 Strong empirical results on **coding**, **logical reasoning**, and **mathematical reasoning** benchmarks  


## 🌈 Overview

<p align="center">
  <img src="assets/overview.png" width="66%">
  <img src="assets/barplot.png" width="33%">
</p>

---

## ⚙️ Step 1: Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/vishnutez/egspo-dllm-rl.git
cd egspo-dllm-rl
conda env create -f environment.yml
conda activate egspo-env
```

---

## 🏋️ Training

Configure the required environment variables (e.g., `WANDB_API_KEY`, `HF_HOME`, etc.) in **epsa/train.sh**


We provide an **`sbatch` script** for running experiments on a cluster. The script can also be easily adapted to a standard `.sh` file if needed.

Run training with:

```bash
sbatch epsa/train.sh
```

Unless otherwise specified in the paper, the **default parameters in `epsa/train.sh` correspond to the configurations used in our experiments**.
