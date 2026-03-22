import torch
from typing import Any, Union
import warnings
import torch.nn.functional as F
from torch import nn
from accelerate.utils import gather, gather_object
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_rich_available
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import print_prompt_completions_sample
import wandb
import os

from egspo_trainer import EGSPOTrainer


class AdaEGSPOTrainer(EGSPOTrainer):
    """
    Adaptive EGSPO Trainer that computes the number of gradient steps per rollout
    batch adaptively based on the entropy of the generated token distributions.

    For a given alpha in (0, 1], K is the minimum number of (highest-entropy) diffusion
    steps needed so that their cumulative entropy covers at least alpha of the total
    entropy across the trajectory.  K is computed per prompt and then max-pooled over
    the batch.  This K replaces the fixed logps_eval_num_steps in _prepare_inputs and
    downstream tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialise to the configured default; will be updated after every generation.
        self._current_K = self.args.logps_eval_num_steps

    # ------------------------------------------------------------------
    # Core: adaptive K computation
    # ------------------------------------------------------------------

    def _compute_adaptive_K(self, unmasked_prob_distributions: torch.Tensor) -> int:
        """
        Compute the adaptive number of gradient steps K from the entropy of the
        probability distributions collected during generation.

        Args:
            unmasked_prob_distributions: (batch_size, diffusion_steps, num_tokens, vocab_size)

        Returns:
            K (int): max over batch of the per-prompt K_i values, clamped to [1, diffusion_steps].
        """
        EPS = 1e-8
        batch_size, diffusion_steps, num_tokens, vocab_size = unmasked_prob_distributions.shape

        # Per-step mean entropy H_t^i  shape: (batch_size, diffusion_steps)
        entropy = -(
            unmasked_prob_distributions * torch.log(unmasked_prob_distributions + EPS)
        ).sum(dim=-1)  # (batch_size, diffusion_steps, num_tokens)
        entropy = entropy.mean(dim=-1)  # (batch_size, diffusion_steps)

        K_values = []
        alpha = self.args.alpha_entropy
        for i in range(batch_size):
            H = entropy[i]  # (diffusion_steps,)
            total_H = H.sum()

            if total_H == 0:
                K_values.append(1)
                continue

            # Sort descending to find minimum k covering alpha fraction
            sorted_H, _ = torch.sort(H, descending=True)
            cumsum_H = torch.cumsum(sorted_H, dim=0)  # (diffusion_steps,)

            # K_i = number of elements in the cumsum needed to reach alpha * total
            threshold = alpha * total_H
            # All positions where cumsum < threshold still need more steps
            K_i = int((cumsum_H < threshold).sum().item()) + 1
            K_i = max(1, min(K_i, diffusion_steps))
            K_values.append(K_i)

        K = max(K_values)
        print(f"[AdaEGSPO] alpha={alpha:.2f}  K_i per prompt={K_values}  K={K}", flush=True)
        return K

    # ------------------------------------------------------------------
    # Override: use self._current_K as the gradient-step period
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            # Regenerate when we have consumed K * num_gradient_steps gradient steps
            period = self._current_K * self.args.num_gradient_steps
            if self._step % period == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._cached_inputs = inputs
            else:
                inputs = self._cached_inputs
            self._step += 1
        return inputs

    # ------------------------------------------------------------------
    # Override: compute_loss – cycle through K time steps (not fixed logps_eval_num_steps)
    # ------------------------------------------------------------------

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        trajectory_ids = inputs["trajectory_ids"]
        eval_time_steps = inputs["eval_time_steps"]

        # Cycle index within the adaptive K steps
        eval_time_step_idx = self._step % self._current_K

        if self.args.logps_eval_mode == "unbiased":
            if eval_time_steps is None:
                raise ValueError("eval_time_steps cannot be None when logps_eval_mode is unbiased")
            per_token_logps, all_tokens_logps = self._get_per_token_logps_unbiased(
                model,
                trajectory_ids,
                eval_time_steps=eval_time_steps[:, eval_time_step_idx].unsqueeze(1),
                get_all_tokens_logps=True,
            )
            per_token_logps = per_token_logps.squeeze(0)
            all_tokens_logps = all_tokens_logps.squeeze(0)
        else:
            raise ValueError(f"Invalid logps_eval_mode: {self.args.logps_eval_mode}")

        if self.beta != 0.0:
            ref_all_tokens_logps = inputs["ref_all_tokens_logps"][eval_time_step_idx]
            ref_per_token_logps = inputs["ref_per_token_logps"][eval_time_step_idx]
            if self.args.logps_aggregation_mode == "sum":
                exact_kl = (
                    (torch.exp(all_tokens_logps) * (all_tokens_logps - ref_all_tokens_logps))
                    .sum(dim=-1)
                ).sum(dim=-1)
            elif self.args.logps_aggregation_mode == "mean":
                exact_kl = (
                    (torch.exp(all_tokens_logps) * (all_tokens_logps - ref_all_tokens_logps))
                    .mean(dim=-1)
                ).mean(dim=-1)
            else:
                raise ValueError(f"Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}")
            k3_estimate_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        else:
            print("beta = 0.0, so no kl term is computed", flush=True)

        advantages = inputs["advantages_per_step"][:, eval_time_step_idx]

        old_per_token_logps = (
            inputs["old_per_token_logps"][eval_time_step_idx]
            if self._current_K > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon_high)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            if self.args.use_exact_kl:
                per_token_loss = per_token_loss + self.beta * exact_kl
            else:
                per_token_loss = per_token_loss + self.beta * k3_estimate_kl
        else:
            print("beta = 0.0, so no kl term is computed", flush=True)

        num_tokens_per_diffusion_step = self.args.max_completion_length // self.args.diffusion_steps

        if self.args.logps_aggregation_mode == "sum":
            loss = per_token_loss.mean() / num_tokens_per_diffusion_step
        elif self.args.logps_aggregation_mode == "mean":
            loss = per_token_loss.mean()
        else:
            raise ValueError(f"Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}")
        print("step:", self._step, "loss:", loss, flush=True)

        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            if self.args.use_exact_kl:
                if self.args.logps_aggregation_mode == "sum":
                    mean_kl = exact_kl.mean() / num_tokens_per_diffusion_step
                elif self.args.logps_aggregation_mode == "mean":
                    mean_kl = exact_kl.mean()
                else:
                    raise ValueError(f"Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}")
            else:
                mean_kl = k3_estimate_kl.mean()
            if self.args.logps_aggregation_mode == "sum":
                self._metrics[mode]["exact_kl"].append(
                    self.accelerator.gather_for_metrics(exact_kl.mean() / num_tokens_per_diffusion_step).mean().item()
                )
            elif self.args.logps_aggregation_mode == "mean":
                self._metrics[mode]["exact_kl"].append(
                    self.accelerator.gather_for_metrics(exact_kl.mean()).mean().item()
                )
            else:
                raise ValueError(f"Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}")
            self._metrics[mode]["k3_estimate_kl"].append(
                self.accelerator.gather_for_metrics(k3_estimate_kl.mean()).mean().item()
            )
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )
        else:
            print("beta = 0.0, so no kl term is logged", flush=True)

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = is_clipped.mean()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss

    # ------------------------------------------------------------------
    # Override: _generate_and_score_completions – adaptive K
    # ------------------------------------------------------------------

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        print("inputs keys:", list(inputs[0].keys()), flush=True)
        print("inputs list length:", len(inputs), flush=True)

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        from transformers import Trainer as _Trainer
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = _Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = self.args.generation_batch_size
            trajectory_ids_all = []
            logps_all = []
            unmasked_prob_distributions_all = []
            greedy_completions_all = []
            last_non_eos_steps_all = []

            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]

                (
                    batch_trajectory_ids,
                    batch_logps,
                    batch_unmasked_prob_distributions,
                    batch_greedy_completions,
                    batch_last_non_eos_steps,
                ) = self.generate(
                    model=unwrapped_model,
                    prompt=batch_prompt_ids,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    remasking=self.args.remasking,
                    mask_id=self.args.mask_id,
                    return_metadata=True,
                )

                trajectory_ids_all.append(batch_trajectory_ids.permute(1, 0, 2))
                logps_all.append(batch_logps.permute(1, 0, 2))
                unmasked_prob_distributions_all.append(batch_unmasked_prob_distributions.permute(1, 0, 2, 3))
                greedy_completions_all.append(batch_greedy_completions.permute(1, 0, 2))
                last_non_eos_steps_all.append(batch_last_non_eos_steps)
                del batch_prompt_ids, batch_prompt_mask, batch_trajectory_ids, batch_logps, batch_unmasked_prob_distributions
                torch.cuda.empty_cache()

            trajectory_ids = torch.cat(trajectory_ids_all, dim=0).permute(1, 0, 2)
            logps = torch.cat(logps_all, dim=0)
            unmasked_prob_distributions = torch.cat(unmasked_prob_distributions_all, dim=0)
            greedy_completions = torch.cat(greedy_completions_all, dim=0)
            last_non_eos_steps = torch.cat(last_non_eos_steps_all, dim=0)

        prompt_length = prompt_ids.size(-1)
        prompt_ids = trajectory_ids[:, :, :prompt_length]
        response_trajectory_ids = trajectory_ids[:, :, prompt_length:]

        self.vocab_size = unmasked_prob_distributions.size(-1)

        completion_ids_final = response_trajectory_ids[-1, :, :]

        is_eos = completion_ids_final == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(-1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(-1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # ----------------------------------------------------------------
        # Compute adaptive K from entropy of this batch's distributions
        # ----------------------------------------------------------------
        K = self._compute_adaptive_K(unmasked_prob_distributions)
        self._current_K = K
        self._metrics["train"]["gradient_steps_per_batch"].append(K)

        # ----------------------------------------------------------------
        # Select eval_time_steps – top-K by entropy (per prompt)
        # ----------------------------------------------------------------
        def _get_high_entropy_time_steps(prob_dists, k):
            """
            Args:
                prob_dists: (batch_size, diffusion_steps, num_tokens, vocab_size)
                k: number of steps to select
            Returns:
                (batch_size, k) long tensor
            """
            EPS = 1e-8
            bs = prob_dists.shape[0]
            result = torch.zeros((bs, k), device=device, dtype=torch.long)
            for b in range(bs):
                pd = prob_dists[b]  # (diffusion_steps, num_tokens, vocab_size)
                ent = -(pd * torch.log(pd + EPS)).sum(dim=-1).mean(dim=-1)  # (diffusion_steps,)
                result[b] = torch.topk(ent, k=k, dim=0).indices
            return result

        if self.args.terminate_at_last_non_eos:
            print(f"last_non_eos_steps (batch_size,) = ({last_non_eos_steps})", flush=True)
            print("terminating at last non-EOS steps", flush=True)

            if self.args.logps_eval_mode == "unbiased":
                eval_time_steps = torch.zeros(
                    (last_non_eos_steps.size(0), K), device=device, dtype=torch.long
                )
                for b in range(last_non_eos_steps.size(0)):
                    if self.args.logps_eval_time_steps_mode == "random":
                        eval_time_steps[b] = torch.randint(
                            0, last_non_eos_steps[b] + 1, (K,)
                        ).to(device)
                    elif self.args.logps_eval_time_steps_mode == "uniform":
                        eval_time_steps[b] = torch.linspace(
                            0, last_non_eos_steps[b], K
                        ).long().to(device)
                    elif self.args.logps_eval_time_steps_mode == "high_entropy":
                        sub = unmasked_prob_distributions[b, : last_non_eos_steps[b] + 1]
                        k_capped = min(K, sub.shape[0])
                        chosen = _get_high_entropy_time_steps(sub.unsqueeze(0), k_capped)
                        eval_time_steps[b, :k_capped] = chosen.squeeze(0)
                        if k_capped < K:
                            eval_time_steps[b, k_capped:] = eval_time_steps[b, k_capped - 1]
                    else:
                        raise ValueError(
                            f"Invalid logps_eval_time_steps_mode: {self.args.logps_eval_time_steps_mode}"
                        )
                print(f"eval_time_steps (batch_size, K={K}) = ({eval_time_steps})", flush=True)

        else:
            if self.args.logps_eval_mode == "unbiased":
                if self.args.logps_eval_time_steps_mode == "random":
                    eval_time_steps_1d = torch.randint(0, self.args.diffusion_steps - 1, (K - 1,)).to(device)
                    eval_time_steps_1d = torch.cat(
                        [eval_time_steps_1d, torch.full((1,), self.args.diffusion_steps - 1, device=device)]
                    )
                    batch_size = unmasked_prob_distributions.size(0)
                    eval_time_steps = eval_time_steps_1d.unsqueeze(0).repeat(batch_size, 1)
                elif self.args.logps_eval_time_steps_mode == "uniform":
                    eval_time_steps_1d = torch.linspace(
                        0, self.args.diffusion_steps - 2, K - 1
                    ).long().to(device)
                    eval_time_steps_1d = torch.cat(
                        [eval_time_steps_1d, torch.full((1,), self.args.diffusion_steps - 1, device=device)]
                    )
                    batch_size = unmasked_prob_distributions.size(0)
                    eval_time_steps = eval_time_steps_1d.unsqueeze(0).repeat(batch_size, 1)
                elif self.args.logps_eval_time_steps_mode == "high_entropy":
                    eval_time_steps = _get_high_entropy_time_steps(unmasked_prob_distributions, K)
                else:
                    eval_time_steps = None

        # ----------------------------------------------------------------
        # old_per_token_logps  shape: (K, batch_size)
        # ----------------------------------------------------------------
        old_per_token_logps = torch.zeros(
            (K, unmasked_prob_distributions.shape[0]), device=device, dtype=torch.float32
        )
        for batch_idx in range(unmasked_prob_distributions.shape[0]):
            if self.args.logps_aggregation_mode == "sum":
                old_per_token_logps[:, batch_idx] = logps[batch_idx, eval_time_steps[batch_idx]].sum(dim=-1)
            elif self.args.logps_aggregation_mode == "mean":
                old_per_token_logps[:, batch_idx] = logps[batch_idx, eval_time_steps[batch_idx]].mean(dim=-1)
            else:
                raise ValueError(f"Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}")

        with torch.no_grad():
            if self.beta == 0.0:
                ref_per_token_logps = None
                ref_all_tokens_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    if self.args.logps_eval_mode == "unbiased":
                        ref_per_token_logps, ref_all_tokens_logps = self._get_per_token_logps_unbiased(
                            self.model,
                            trajectory_ids,
                            eval_time_steps=eval_time_steps,
                            get_all_tokens_logps=True,
                        )
                    else:
                        ref_per_token_logps = None

        batch_size = unmasked_prob_distributions.shape[0]

        completion_ids = greedy_completions[:, :, prompt_length:]
        len_completion_ids = completion_ids.size(-1)
        selected_completion_ids = torch.zeros(
            (batch_size, K + 1, len_completion_ids), device=device, dtype=torch.long
        )

        for b in range(batch_size):
            for t in range(K):
                selected_completion_ids[b, t] = completion_ids[b, eval_time_steps[b, t]]
        selected_completion_ids[:, -1] = completion_ids[:, -1]

        print(
            f"selected_completion_ids (batch_size, K+1={K+1}, len_completion_ids) = ({selected_completion_ids.shape})",
            flush=True,
        )

        selected_completion_ids_flattened = selected_completion_ids.reshape(-1, len_completion_ids)
        print(
            f"selected_completion_ids_flattened (batch_size*(K+1), len_completion_ids) = ({selected_completion_ids_flattened.shape})",
            flush=True,
        )

        print("prompts:", len(prompts), flush=True)

        prompts_full = [prompt for prompt in prompts for _ in range(K + 1)]

        completions_text = self.processing_class.batch_decode(
            selected_completion_ids_flattened, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts_full, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
            print("in conversational mode", flush=True)
        else:
            print("in non-conversational mode", flush=True)
            completions = completions_text

        print(f"prompts_full: {len(prompts_full)}", flush=True)
        print(f"completions_text: {len(completions_text)}", flush=True)
        print(f"completions: {len(completions)}", flush=True)

        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {
                    key: [example[key] for example in inputs for _ in range(K + 1)]
                    for key in keys
                }
                if reward_func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                output_reward_func = reward_func(
                    prompts=prompts_full,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts_full[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        print(f"rewards_per_func (batch_size*(K+1), num_reward_funcs) = ({rewards_per_func.shape})", flush=True)

        rewards_per_func = rewards_per_func.reshape(-1, K + 1, len(self.reward_funcs))

        final_rewards_per_func = rewards_per_func[:, -1]
        final_rewards_per_func_all_devices = gather(final_rewards_per_func)

        print(
            f"final_rewards_per_func_all_devices (batch_size, num_reward_funcs) = ({final_rewards_per_func_all_devices})",
            flush=True,
        )

        if self.args.correctness_step_reward_only:
            custom_reward_weights = torch.zeros((K + 1, len(self.reward_funcs)))
            custom_reward_weights[:, -1] = 1.0
            custom_reward_weights[-1, :] = 1.0
            custom_reward_weights = custom_reward_weights.to(device).unsqueeze(0)
            print("only using correctness reward for the intermediate steps", flush=True)
        else:
            custom_reward_weights = self.reward_weights.to(device).unsqueeze(0).unsqueeze(0)

        rewards_per_step = (rewards_per_func * custom_reward_weights).nansum(dim=2)

        print(
            f"rewards_per_step (per_device_train_batch_size, K+1={K+1}) = ({rewards_per_step.shape})",
            flush=True,
        )

        final_rewards = rewards_per_step[:, -1]
        final_rewards_all_devices = gather(final_rewards)

        print(f"final_rewards_all_devices (batch_size,) = ({final_rewards_all_devices})", flush=True)

        stepwise_advantages = final_rewards.unsqueeze(1) - rewards_per_step[:, :-1]

        if self.args.dynamic_lambda1:
            if self.state.global_step % self.args.lambda1_update_steps == 0:
                self.args.lambda1 = self.args.lambda1 / 2
                print(f"updating lambda1 to {self.args.lambda1}", flush=True)

        if self.args.standard_grpo_returns:
            returns_local_selected = final_rewards.unsqueeze(1)
        else:
            returns_local_selected = final_rewards.unsqueeze(1) + self.args.lambda1 * stepwise_advantages
            if self.args.normalize_returns:
                print("normalizing returns by 1 + lambda1", flush=True)
                returns_local_selected = returns_local_selected / (1 + self.args.lambda1)
            else:
                print("not normalizing returns", flush=True)

        step_reward_ratios = rewards_per_step[:, :-1] / final_rewards.unsqueeze(1)
        step_reward_ratios_all_devices = self.accelerator.gather_for_metrics(step_reward_ratios)
        eval_time_steps_all_devices = self.accelerator.gather_for_metrics(eval_time_steps)

        returns = gather(returns_local_selected)
        batch_size_global, K_check = returns.shape
        returns_grouped = returns.view(-1, self.num_generations, K_check)

        print(
            f"returns_grouped (num_prompts, num_generations, K={K_check}) = ({returns_grouped.shape})",
            flush=True,
        )

        mean_grouped_returns = returns_grouped.mean(dim=1, keepdim=True)
        print(f"mean_grouped_returns = ({mean_grouped_returns})", flush=True)

        advantages_per_step = returns_grouped - mean_grouped_returns
        advantages_per_step = advantages_per_step.view(batch_size_global, K_check)

        print(
            f"advantages_per_step (batch_size, K={K_check}) = ({advantages_per_step.shape})",
            flush=True,
        )

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages_per_step = advantages_per_step[process_slice]

        if self.args.standard_grpo_returns:
            advantages_per_step = advantages_per_step + self.args.lambda1 * stepwise_advantages
            if self.args.normalize_returns:
                print("normalizing advantages by (1 + lambda1)", flush=True)
                advantages_per_step = advantages_per_step / (1 + self.args.lambda1)
            else:
                print("not normalizing advantages", flush=True)

        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            mean_rewards = torch.nanmean(final_rewards_per_func_all_devices[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(final_rewards_all_devices.mean().item())

        for step_idx in range(step_reward_ratios_all_devices.shape[1]):
            self._metrics["eval"][f"ablation/step_rewards_ratio_{step_idx}"].append(
                step_reward_ratios_all_devices[:, step_idx].mean().item()
            )

        for step_idx in range(eval_time_steps_all_devices.shape[1]):
            self._metrics["eval"][f"ablation/time_step_{step_idx}"].append(
                eval_time_steps_all_devices[:, step_idx].float().mean().item()
            )

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = final_rewards_all_devices.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    table = {
                        "step": [str(self.state.global_step)] * len(final_rewards_all_devices),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": final_rewards_all_devices.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log(
                        {"completions": wandb.Table(dataframe=df)},
                        step=self.state.global_step,
                    )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "eval_time_steps": eval_time_steps,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "ref_all_tokens_logps": ref_all_tokens_logps,
            "advantages_per_step": advantages_per_step,
            "trajectory_ids": trajectory_ids,
        }
