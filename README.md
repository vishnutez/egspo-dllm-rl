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


We provide an multi-node **`sbatch` script** for running experiments on a cluster. The script can also be easily adapted to a standard `.sh` file if needed.

Run training with:

```bash
sbatch egspo/train.sh
```

Unless otherwise specified in the paper, the **default parameters in `epsa/train.sh` correspond to the configurations used in our experiments**.

---
## 📊 Evaluation

### ⚡ Step 1: Generate completions

Before running evaluation, update the required fields in:

```bash
eval/eval_checkpoints.sh
```

In particular, set the following variables:

- `CHECKPOINT_DIR` — directory containing the trained model checkpoints to evaluate  
- `OUTPUT_DIR` — directory where generated completions will be saved  
- `TASKS` — evaluation task(s) (e.g., `gsm8k`, `sudoku`, etc.)  
- `GEN_LENGTHS` — generation lengths to evaluate  
- `<YOUR_CONDA_ENV_NAME>` — your conda environment name  
- `<YOUR_HF_HOME_DIR>` *(optional)* — Hugging Face cache directory (remove if using the default)

Then run:

```bash
bash eval/eval_checkpoints.sh
```

This step generates **model completions for the test prompts** using the selected checkpoints.  
The script also **parses predicted answers from model outputs** and **extracts ground-truth answers from the dataset**, preparing them for evaluation.



### 📈 Step 2: Compute metrics

Modify the following fields in:

```bash
eval/get_and_save_metrics.py
```

- `task`
- `checkpoint_dir`
- `generated_lengths`

Using the completions generated in the previous step, this script **computes evaluation metrics by comparing predicted answers with ground-truth answers**.  
The results are then **saved as `.json` files** for each evaluated checkpoint.

---

## 🙏 Acknowledgement

Our implementation builds upon the codebase from the **d1 paper**: https://github.com/dllm-reasoning/d1/tree/main/diffu-grpo

We thank the authors for making their implementation publicly available, which helped facilitate this work.

---

## 📬 Contact

If you have any questions or concerns, feel free to contact us:

- fatemehdoudi@tamu.edu  
- vishnukunde@tamu.edu  

You can also open an issue in this repository.

---

## 📜 Citation

If you find our codebase or research useful, please consider citing our work:

```bibtex
@article{kunde2026reinforcement,
  title={Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages},
  author={Kunde, Vishnu Teja and Doudi, Fatemeh and Farahbakhsh, Mahdi and Kalathil, Dileep and Narayanan, Krishna and Chamberland, Jean-Francois},
  year={2026}
}
```
