# Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages

Official repository for the paper **“Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages”** by **Vishnu Teja Kunde, Fatemeh Doudi, Mahdi Farahbakhsh, Dileep Kalathil, Krishna Narayanan, and Jean‑Francois Chamberland**. fileciteturn0file0

This work studies reinforcement learning for diffusion language models (DLMs) by formulating diffusion-based sequence generation as a finite-horizon Markov decision process over denoising steps, deriving an exact policy gradient with stepwise advantages, and introducing practical estimators based on entropy-guided step selection and one-step denoising rewards. fileciteturn0file0

## Highlights

- **Diffusion-MDP formalism** for RL fine-tuning of diffusion LLMs. fileciteturn0file0
- **Exact policy gradient** that decomposes across denoising steps with a principled notion of **stepwise advantages**. fileciteturn0file0
- **Entropy-Guided Stepwise Policy Optimization (EGSPO)** for selecting the most informative denoising steps under a fixed compute budget. fileciteturn0file0
- **EGSPO-SA**, which augments EGSPO with lightweight **stepwise advantage estimation** using one-step denoising completions. fileciteturn0file0
- Strong empirical results on **coding** and **logical reasoning** benchmarks, with competitive performance on **mathematical reasoning**. fileciteturn0file0

## Paper

**Title:** Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages  
**Authors:** Vishnu Teja Kunde, Fatemeh Doudi, Mahdi Farahbakhsh, Dileep Kalathil, Krishna Narayanan, Jean‑Francois Chamberland  
**Date:** March 12, 2026 fileciteturn0file0

### Abstract

> Reinforcement learning (RL) has been effective for post-training autoregressive language models, but extending these methods to diffusion language models is challenging due to intractable sequence-level likelihoods. This work derives an exact, unbiased policy gradient over denoising steps, introduces entropy-guided step selection, and proposes stepwise advantage estimation based on one-step denoising rewards. Experiments on coding and reasoning benchmarks show strong performance over existing RL post-training methods for diffusion LLMs. fileciteturn0file0

## Methods

### EGSPO
Entropy-Guided Stepwise Policy Optimization selects a subset of denoising steps with the highest policy entropy, focusing training on the most informative parts of the diffusion trajectory. fileciteturn0file0

### EGSPO-SA
EGSPO-SA extends EGSPO with stepwise advantage estimation. Instead of relying only on sequence-level reward, it uses a one-step denoising completion to estimate intermediate value signals without requiring a separate value model or expensive multi-step rollouts. fileciteturn0file0

## Results Summary

According to the paper, the proposed methods:
- outperform existing baselines on **coding** and **logical reasoning** tasks,
- remain competitive on **mathematical reasoning**,
- and improve compute efficiency by concentrating updates on informative denoising steps. fileciteturn0file0

Benchmarks discussed in the paper include:
- **Sudoku**
- **Countdown**
- **GSM8K**
- **MATH500**
- **HumanEval**
- **MBPP** fileciteturn0file0

## Repository Structure

From the repository layout shown in the provided screenshot, the project currently contains:

```text
dataset/
epsa/
eval/
.gitignore
README.md
environment.yml
```

The repository description states that it is the official implementation for entropy-guided policy-gradient RL fine-tuning of diffusion LLMs. The repository README shown in the screenshot uses the project name **EGSPO-SA**.

## Setup

The repository includes an `environment.yml`, so the expected setup is:

```bash
conda env create -f environment.yml
conda activate epsa
```

If the created environment name differs, check `environment.yml` and activate the exact environment name defined there.

## Usage

Training and evaluation code appear to live under:
- `epsa/` for the main algorithm implementation
- `eval/` for evaluation utilities
- `dataset/` for task data or preprocessing assets

Because exact command-line entrypoints are not visible in the provided materials, update this section with the concrete commands used in your repo, for example:

```bash
# training
python -m epsa.train ...

# evaluation
python -m eval.run ...
```

## Citation

```bibtex
@article{kunde2026reinforcement,
  title={Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages},
  author={Kunde, Vishnu Teja and Doudi, Fatemeh and Farahbakhsh, Mahdi and Kalathil, Dileep and Narayanan, Krishna and Chamberland, Jean-Francois},
  year={2026},
  note={Preprint}
}
```

## Acknowledgment

This work used advanced computing resources provided by **Texas A&M High Performance Research Computing** and was supported in part by the **National Science Foundation** and **U.S. Army DEVCOM**, as described in the paper acknowledgments. fileciteturn0file0

## Repository Link

GitHub repository shown in the provided screenshot:

`github.com/vishnutez/epsa-dllm-rl`

---

If you plan to use this repository publicly, it would also help to add:
- a short **Quick Start**
- exact **training/evaluation commands**
- expected **data format**
- **checkpoint** information
- and a **license** section
