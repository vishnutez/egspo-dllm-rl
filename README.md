## Paper Overview

**Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages** studies how to apply reinforcement learning to **diffusion language models (dLLMs)**. Unlike autoregressive LLMs, diffusion models generate sequences through an iterative denoising process, which makes standard RL fine-tuning techniques difficult to apply.

This work formulates the denoising trajectory as a **finite-horizon Markov decision process** and derives a policy-gradient objective that decomposes across denoising steps. Building on this formulation, we introduce **Entropy-Guided Stepwise Policy Optimization (EGSPO)**, which focuses training on the most informative denoising steps based on policy entropy.

We further propose **EGSPO-SA**, a practical variant that estimates **stepwise advantages** using lightweight one-step denoising completions, enabling efficient RL fine-tuning without requiring a separate value network or expensive multi-step rollouts.

Experiments on reasoning and coding benchmarks demonstrate improved training efficiency and competitive performance for RL-aligned diffusion LLMs.
