"""
CrStReL — Critical Step Reinforcement Learning trainer.

Extends EGSPOTrainer with the `critical_confidence` eval step selection strategy:

  - After computing GRPO-centered final-reward advantages, completions are split into:
      positive_completions  S+ = { j : A_j > 0 }
      negative_completions  S- = { j : A_j <= 0 }
  - For positive completions: select the `logps_eval_num_steps` diffusion steps
    with the *lowest* generation log-probs (least confident / most uncertain decisions).
  - For negative completions: select the `logps_eval_num_steps` diffusion steps
    with the *highest* generation log-probs (most confident / certain decisions).
  - The policy gradient at the selected steps uses the GRPO final-reward advantage
    (uniform across all selected steps for a given completion).
"""

import torch
from typing import Any, Union
from torch import nn
from accelerate.utils import gather, gather_object
from trl.models import unwrap_model_for_generation
from trl.extras.profiling import profiling_context
from trl.import_utils import is_rich_available
from trl.data_utils import is_conversational
from trl.trainer.utils import print_prompt_completions_sample
import wandb
import os

from egspo_trainer import EGSPOTrainer


class CrStReLTrainer(EGSPOTrainer):
    """
    Critical Step Reinforcement Learning (CrStReL) trainer.

    Inherits all generation, log-prob, and loss machinery from EGSPOTrainer.
    Adds the `critical_confidence` eval step selection mode and overrides
    `_generate_and_score_completions` to implement the two-phase flow:
    (1) score final completions → center advantages,
    (2) select critical steps per completion based on advantage sign and model confidence.
    """

    # ------------------------------------------------------------------
    # New method: critical confidence step selection
    # ------------------------------------------------------------------

    def _critical_confidence_steps(self, gen_logps, local_final_adv, n_eval):
        """
        Select diffusion steps based on advantage sign and generation-time confidence.

        For positive_completions (A > 0): pick the n_eval steps with the *lowest*
        mean gen_logps — the steps where the model was least confident, i.e. the
        most uncertain decisions that still led to a good outcome.

        For negative_completions (A <= 0): pick the n_eval steps with the *highest*
        mean gen_logps — the steps where the model was most confident, i.e. the
        decisions it was sure about that led to a bad outcome.

        Args:
            gen_logps:        (bs, T, tokens_per_step) — log-probs from generation.
            local_final_adv:  (bs,) — GRPO-centered final-reward advantages (local device).
            n_eval:           number of steps to select per completion.
        Returns:
            eval_steps: (bs, n_eval)
        """
        bs, T, _ = gen_logps.shape
        device = gen_logps.device

        # Average over tokens to get a scalar confidence score per step
        step_confidence = gen_logps.mean(dim=-1)  # (bs, T)

        positive_completions = local_final_adv >= 0   # (bs,) bool
        negative_completions = ~positive_completions  # (bs,) bool

        eval_steps = torch.zeros((bs, n_eval), device=device, dtype=torch.long)

        if positive_completions.any():
            # Least confident steps: reinforce uncertain decisions that led to success
            eval_steps[positive_completions] = torch.topk(
                step_confidence[positive_completions], k=n_eval, largest=False
            ).indices

        if negative_completions.any():
            # Most confident steps: penalize confident decisions that led to failure
            eval_steps[negative_completions] = torch.topk(
                step_confidence[negative_completions], k=n_eval, largest=True
            ).indices

        print(
            f'positive_completions: {positive_completions.sum().item()}, '
            f'negative_completions: {negative_completions.sum().item()}',
            flush=True,
        )
        return eval_steps

    # ------------------------------------------------------------------
    # Override: generate-and-score with critical_confidence branching
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Fall back to EGSPOTrainer for all other eval_step_selection modes
        if self.args.logps_eval_step_selection != 'critical_confidence':
            return super()._generate_and_score_completions(inputs)

        # ── CrStReL: critical_confidence flow ─────────────────────────
        device = self.accelerator.device
        print('inputs keys: ', list(inputs[0].keys()), flush=True)

        prompts, prompts_text, prompt_ids, prompt_mask = self._tokenize_prompts(inputs)
        prompt_len = prompt_ids.size(-1)

        # Phase 1: generate full diffusion trajectories
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            traj, gen_logps, unmask_probs, greedy_completions, last_non_eos_steps = \
                self._run_generation(unwrapped_model, prompt_ids, prompt_mask)

        self.vocab_size = unmask_probs.size(-1)
        bs = gen_logps.shape[0]
        n_eval = self.args.logps_eval_num_steps

        # Build completion mask up to first EOS
        final_completions = traj[-1, :, prompt_len:]
        is_eos = final_completions == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(-1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(-1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        # Phase 2: score FINAL completions only to obtain rewards and advantages
        completion_ids = greedy_completions[:, :, prompt_len:]  # (bs, T+1, comp_len)
        final_comp = completion_ids[:, -1]                       # (bs, comp_len)

        completions_text = self.processing_class.batch_decode(final_comp, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
            print('in conversational mode', flush=True)
        else:
            print('in non-conversational mode', flush=True)
            completions = completions_text

        func_rewards = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            func_name = (
                f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                if isinstance(reward_func, nn.Module) else reward_func.__name__
            )
            with profiling_context(self, func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [ex[key] for ex in inputs] for key in keys}
                if func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                raw_rewards = reward_func(
                    prompts=prompts,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                raw_rewards = [r if r is not None else torch.nan for r in raw_rewards]
                func_rewards[:, i] = torch.tensor(raw_rewards, dtype=torch.float32, device=device)

        final_func_rewards = func_rewards                         # (bs, num_reward_funcs)
        final_func_rewards_all = gather(final_func_rewards)
        final_rewards = (func_rewards * self.reward_weights.to(device)).nansum(dim=1)  # (bs,)
        final_rewards_all = gather(final_rewards)

        # Phase 3: compute GRPO-centered advantages across all devices
        # Change 1 — advantages are computed for all steps; since intermediate
        # rewards are not scored, the advantage at every time step equals the
        # GRPO final-reward advantage for that completion.
        returns = gather(final_rewards)                           # (total_bs,)
        grouped_returns = returns.view(-1, self.num_generations)  # (num_prompts, G)
        mean_returns = grouped_returns.mean(dim=1, keepdim=True)  # (num_prompts, 1)
        final_adv_global = (grouped_returns - mean_returns).view(-1)  # (total_bs,)

        proc_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        local_final_adv = final_adv_global[proc_slice]            # (bs,)

        print(f'local_final_adv: {local_final_adv}', flush=True)

        # Phase 4: select critical steps (Change 2 — critical_confidence strategy)
        eval_steps = self._critical_confidence_steps(gen_logps, local_final_adv, n_eval)
        print(f'eval_steps (bs, n_eval) = {eval_steps}', flush=True)

        # Phase 5: old log-probs from generation at selected steps
        old_logps = torch.zeros((n_eval, bs), device=device, dtype=torch.float32)
        for b in range(bs):
            if self.args.logps_aggregation == 'sum':
                old_logps[:, b] = gen_logps[b, eval_steps[b]].sum(dim=-1)
            elif self.args.logps_aggregation == 'mean':
                old_logps[:, b] = gen_logps[b, eval_steps[b]].mean(dim=-1)
            else:
                raise ValueError(f'Invalid logps_aggregation: {self.args.logps_aggregation}')

        # Phase 6: reference model log-probs
        ref_logps, ref_all_logps = self._compute_ref_logps(traj, eval_steps)

        # Phase 7: step_advs = final-reward advantage applied uniformly to all eval steps
        # (Change 1 — advantages for all time steps are the GRPO final-reward advantage)
        step_advs = local_final_adv.unsqueeze(1).expand(-1, n_eval).clone()  # (bs, n_eval)

        # Logging
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        for i, reward_func in enumerate(self.reward_funcs):
            func_name = (
                reward_func.config._name_or_path.split("/")[-1]
                if isinstance(reward_func, nn.Module) else reward_func.__name__
            )
            self._metrics[mode][f"rewards/{func_name}"].append(
                torch.nanmean(final_func_rewards_all[:, i]).item()
            )
        self._metrics[mode]["reward"].append(final_rewards_all.mean().item())

        eval_steps_all = self.accelerator.gather_for_metrics(eval_steps)
        for t in range(eval_steps_all.shape[1]):
            self._metrics["eval"][f"ablation/time_step_{t}"].append(
                eval_steps_all[:, t].float().mean().item()
            )

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log, completions_to_log,
                        final_rewards_all.tolist(), self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd
                    df = pd.DataFrame({
                        "step": [str(self.state.global_step)] * len(final_rewards_all),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": final_rewards_all.tolist(),
                    })
                    wandb.log({"completions": wandb.Table(dataframe=df)}, step=self.state.global_step)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "eval_steps": eval_steps,
            "old_logps": old_logps,
            "ref_logps": ref_logps,
            "ref_all_logps": ref_all_logps,
            "step_advs": step_advs,
            "traj": traj,
        }
