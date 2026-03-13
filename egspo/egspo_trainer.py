import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import wandb
from collections import defaultdict
import os

if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class EGSPOTrainer(GRPOTrainer):
    """
    Group Relative Policy Optimization (GRPO) Trainer for Diffusion Language Models.

    This class extends the GRPOTrainer to adapt it for masked diffusion language models,
    implementing efficient policy gradient estimation through conditional probabilities
    with masked tokens.

    Key features:
    - Random masking for improved robustness in multiple policy optimization updates
    - Efficient computation of per-token log probabilities for diffusion models
    - Specialized generation process for diffusion models with iterative denoising
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self._step = getattr(self.state, "global_step", 0)


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        trajectory_ids = inputs["trajectory_ids"]  # (per_device_train_batch_size, diffusion_steps+1, seq_len)
        eval_time_steps = inputs["eval_time_steps"]

        eval_time_step_idx = self._step % self.args.logps_eval_num_steps

        if self.args.logps_eval_mode == 'unbiased':
            if eval_time_steps is None:
                raise ValueError(f'eval_time_steps cannot be None when logps_eval_mode is unbiased')
            per_token_logps, all_tokens_logps = self._get_per_token_logps_unbiased(
                model, trajectory_ids, eval_time_steps=eval_time_steps[:, eval_time_step_idx].unsqueeze(1), get_all_tokens_logps=True
            ) # returns (1, bs) squeezed to (bs,)  input: eval_time_steps.shape: (bs, 1)
            per_token_logps = per_token_logps.squeeze(0) # (bs,)
            all_tokens_logps = all_tokens_logps.squeeze(0) # (bs, num_tokens_per_diffusion_step, vocab_size)
        else:
            raise ValueError(f'Invalid logps_eval_mode: {self.args.logps_eval_mode}')

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_all_tokens_logps = inputs["ref_all_tokens_logps"][eval_time_step_idx] # (bs, num_tokens_per_diffusion_step, vocab_size)
            ref_per_token_logps = inputs["ref_per_token_logps"][eval_time_step_idx] # (bs,)
            if self.args.logps_aggregation_mode == 'sum':
                exact_kl = ((torch.exp(all_tokens_logps) * (all_tokens_logps - ref_all_tokens_logps)).sum(dim=-1)).sum(dim=-1) # (bs,)
            elif self.args.logps_aggregation_mode == 'mean':
                exact_kl = ((torch.exp(all_tokens_logps) * (all_tokens_logps - ref_all_tokens_logps)).mean(dim=-1)).mean(dim=-1) # (bs,)
            else:
                raise ValueError(f'Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}')
            k3_estimate_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            ) # (bs,) using prod distribution over unmasked tokens in that step
        
        else:
            print(f'beta = 0.0, so no kl term is computed', flush=True)

        # Compute the loss
        advantages = inputs["advantages_per_step"][:, eval_time_step_idx]

        old_per_token_logps = inputs["old_per_token_logps"][eval_time_step_idx] if self.args.logps_eval_num_steps > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps) # [per_device_train_batch_size,]
        coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon_high)
        per_token_loss1 = coef_1 * advantages # [per_device_train_batch_size,]
        per_token_loss2 = coef_2 * advantages # [per_device_train_batch_size,]
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) # [per_device_train_batch_size,]
        if self.beta != 0.0:
            if self.args.use_exact_kl:
                per_token_loss = per_token_loss + self.beta * exact_kl
            else:
                per_token_loss = per_token_loss + self.beta * k3_estimate_kl
        else:
            print(f'beta = 0.0, so no kl term is computed', flush=True)

        num_tokens_per_diffusion_step = self.args.max_completion_length // self.args.diffusion_steps
            
        # Divided by num_tokens_per_diffusion_step due to prod distribution over unmasked tokens in that step to get the loss per token
        if self.args.logps_aggregation_mode == 'sum':
            loss = per_token_loss.mean() / num_tokens_per_diffusion_step
        elif self.args.logps_aggregation_mode == 'mean':
            loss = per_token_loss.mean()
        else:
            raise ValueError(f'Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}')
        print('step: ', self._step, 'loss: ', loss, flush=True)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            if self.args.use_exact_kl:
                if self.args.logps_aggregation_mode == 'sum':
                    mean_kl = exact_kl.mean() / num_tokens_per_diffusion_step
                elif self.args.logps_aggregation_mode == 'mean':
                    mean_kl = exact_kl.mean()
                else:
                    raise ValueError(f'Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}')
            else:
                mean_kl = k3_estimate_kl.mean()
            if self.args.logps_aggregation_mode == 'sum':
                self._metrics[mode]["exact_kl"].append(self.accelerator.gather_for_metrics(exact_kl.mean() / num_tokens_per_diffusion_step).mean().item())
            elif self.args.logps_aggregation_mode == 'mean':
                self._metrics[mode]["exact_kl"].append(self.accelerator.gather_for_metrics(exact_kl.mean()).mean().item())
            else:
                raise ValueError(f'Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}')
            self._metrics[mode]["k3_estimate_kl"].append(self.accelerator.gather_for_metrics(k3_estimate_kl.mean()).mean().item())
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        else:
            print(f'beta = 0.0, so no kl term is logged', flush=True)
        
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = is_clipped.mean()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss


    def add_gumbel_noise(self, logits, temperature, dtype):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if temperature == 0.0:
            return logits  # Skip noise when temperature is 0
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    @torch.no_grad()
    def generate(
        self,
        model,
        prompt,
        steps=128,
        gen_length=128,
        block_length=128,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        return_metadata=False,
    ):
        """generation code adopted from llada (https://github.com/ML-GSAI/LLaDA)"""
        with torch.amp.autocast(device_type='cuda', enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
            x[:, : prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length

            # Adjust steps if needed
            steps_per_block = max(1, steps // num_blocks)

            trajectory = []
            trajectory.append(x.clone())
            
            # masked_positions = []

            is_eos_decoded = torch.full((bs, steps), False, dtype=torch.bool, device=x.device)

            if return_metadata:
                logps = []
                unmasked_prob_distributions = []
                greedy_completions = []

            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = x[:, start_idx:end_idx] == mask_id
                num_transfer_tokens = self.get_num_transfer_tokens(block_mask_index, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id

                    time_step = num_block * steps_per_block + i

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            # Handle classifier-free guidance more efficiently
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)

                                # Get logits in a single forward pass
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            # Apply Gumbel noise for sampling
                            logits_with_noise = self.add_gumbel_noise(
                                logits, temperature=temperature, dtype=dtype
                            )
                            x0 = torch.argmax(logits_with_noise, dim=-1)

                            if return_metadata:
                                x0_greedy = torch.argmax(logits, dim=-1)
                                x0_greedy[~mask_index] = x[~mask_index] # replace the decoded tokens with the previously decoded tokens
                                greedy_completions.append(x0_greedy.clone())
                            
                            del logits_with_noise

                            # Handle remasking strategy
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            # Ensure we don't process tokens beyond the current block
                            x0_p[:, end_idx:] = -np.inf

                            # Update masked tokens
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            # Select tokens to transfer based on confidence
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                num_tokens = num_transfer_tokens[j, i].item()
                                if num_tokens > 0:
                                    _, select_index = torch.topk(confidence[j], k=num_tokens)
                                    transfer_index[j, select_index] = True

                            x[transfer_index] = x0[transfer_index]
                            decoded_tokens = x0[transfer_index] # (num_tokens,)
                            # reshape decoded_tokens to (bs, num_tokens)
                            decoded_tokens = decoded_tokens.reshape(bs, -1)
                            # print('decoded_tokens shape: ', decoded_tokens.shape, flush=True)
                            is_eos_decoded[:, time_step] = (decoded_tokens == self.processing_class.eos_token_id).all(dim=1)
                            del confidence

                            trajectory.append(x.clone())
                            if return_metadata:
                                # Compute probabilities for the transferred tokens
                                if remasking == "low_confidence":
                                    prob_distribution = p.to(dtype)  # Reuse the probability distribution computed by low confidence strategy  # (batch_size, seq_len, vocab_size)
                                else:
                                    prob_distribution = F.softmax(logits.to(dtype), dim=-1)  # (batch_size, seq_len, vocab_size)

                                # Pick probability distributions at locations given by transfer_index in seq_len dimension (dim=1)
                                # transfer_index: (batch_size, seq_len) boolean tensor
                                # prob_distribution: (batch_size, seq_len, vocab_size)
                                # Result: (batch_size, num_tokens, vocab_size) - prob distributions for transferred tokens per batch
                                batch_unmasked_prob_distributions = []
                                batch_unmasked_logps = []
                                for j in range(bs):
                                    # Get indices where transfer_index[j] is True
                                    unmasked_indices = torch.where(transfer_index[j])[0]  # (num_tokens,)
                                    unmasked_tokens = x0[j, unmasked_indices]  # (num_tokens,)
                                    # Gather probability distributions at those indices
                                    batch_unmasked_prob_distribution = prob_distribution[j, unmasked_indices]  # (num_tokens, vocab_size)
                                    
                                    # Select probabilities for the actual tokens from vocab_size dimension
                                    token_probs = batch_unmasked_prob_distribution.gather(dim=1, index=unmasked_tokens.unsqueeze(1)).squeeze(1)  # (num_tokens,)
                                    
                                    batch_unmasked_prob_distributions.append(batch_unmasked_prob_distribution)
                                    batch_unmasked_logps.append(torch.log(token_probs))

                                del x0
                                unmasked_prob_distributions.append(torch.stack(batch_unmasked_prob_distributions, dim=0))  # (batch_size, num_tokens, vocab_size)
                                logps.append(torch.stack(batch_unmasked_logps, dim=0))  # (batch_size, num_tokens,)

            # Make the trajectory a tensor
            trajectory = torch.stack(trajectory, dim=0) # (diffusion_steps+1, batch_size, seq_len)
            greedy_completions.append(trajectory[-1, :, :])  # add the final state of the trajectory to the greedy completions

            last_non_eos_steps = steps - 1 - is_eos_decoded.flip(dims=[1]).int().argmin(dim=1) # (bs,) last time step when EOS was NOT decoded
            print('last_non_eos_steps: ', last_non_eos_steps, flush=True)

            if return_metadata:
                logps = torch.stack(logps, dim=0) # (diffusion_steps, batch_size, num_tokens,)
                unmasked_prob_distributions = torch.stack(unmasked_prob_distributions, dim=0) # (diffusion_steps, batch_size, num_tokens, vocab_size)
                greedy_completions = torch.stack(greedy_completions, dim=0) # (diffusion_steps+1, batch_size, seq_len)
                return trajectory, logps, unmasked_prob_distributions, greedy_completions, last_non_eos_steps
            else:
                return trajectory

            


    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        # Create a random matrix to decide whether each prompt token is masked
        random_matrix = torch.rand((b, l), device=batch.device)

        # For prompt tokens: mask if random_matrix < t_p
        # For completion tokens: always mask
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index  # all completion tokens are masked
        is_mask = is_mask_prompt | is_mask_completion

        # Create a noisy (masked) batch
        noisy_batch = torch.where(is_mask, mask_id, batch)

        # Build p_mask, the probability that each token is masked under this scheme
        #   - p_mask[i, j] = t_p[i] if it's a prompt token
        #   - p_mask[i, j] = 1      if it's a completion token
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),  # prompt token probability
            torch.ones_like(t_p).unsqueeze(1),  # completion token probability
        )

        return noisy_batch, p_mask

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        input = batch
        logits = model(input).logits

        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        Precompute the number of tokens to transition at each step.
        Optimized to be more efficient.
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        # Create tensor once and modify in-place
        num_transfer_tokens = base.expand(-1, steps).clone()

        # Handle remainder more efficiently
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1

        return num_transfer_tokens.to(torch.int64)
   
    
    # Loop implementation of _get_per_token_logps function
    def _get_per_token_logps_unbiased(self, model, trajectory_ids, eval_time_steps, get_all_tokens_logps=False):
        """
        Calculate per-token log probabilities.
        
        Args:
            eval_time_steps: If 1D [num_steps], same time steps for all batches.
                           If 2D [batch_size, num_steps], different time steps per batch.
        """

        print('Unbiased GRPO: _get_per_token_logps_unbiased', flush=True)

        # masked_positions: [diffusion_steps, batch_size, seq_len]
        _, batch_size, seq_len = trajectory_ids.size()     
        device = trajectory_ids.device
        final_state = trajectory_ids[-1, :, :] # [batch_size, seq_len]

        # Convert to tensor if not already
        if not isinstance(eval_time_steps, torch.Tensor):
            eval_time_steps = torch.tensor(eval_time_steps, device=device, dtype=torch.long)

        # Handle 1D case (same eval_time_steps for all batches)
        if eval_time_steps.dim() == 1:
            num_steps = len(eval_time_steps)
            per_token_logps = torch.zeros(num_steps, batch_size, device=device)  # [num_steps, batch_size]

            for i in range(num_steps):
                curr_state = trajectory_ids[eval_time_steps[i], :,  :] # [batch_size, seq_len]
                next_state = trajectory_ids[eval_time_steps[i] + 1, :, :] # [batch_size, seq_len]             

                if self.args.pred_state == 'next':
                    positions = curr_state != next_state # [batch_size, seq_len]
                    targets = next_state[positions] # [positions.sum()]
                elif self.args.pred_state == 'final':
                    positions = curr_state != final_state # [batch_size, seq_len]
                    targets = final_state[positions] # [positions.sum()]
                else:
                    raise ValueError(f'Invalid pred_state: {self.args.pred_state}')

                pred_logits_all = model(curr_state).logits # [batch_size, seq_len, vocab_size]
                pred_logits = pred_logits_all[positions] # [positions.sum(), vocab_size]

                # Compute per-token cross-entropy losses
                per_token_losses = F.cross_entropy(pred_logits, targets, reduction="none")  # [positions.sum()]

                # Aggregate per-token losses back to per-batch losses
                batch_indices_all = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
                batch_indices = batch_indices_all[positions]  # [positions.sum()]

                # Sum the losses
                summed_losses = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                    0, batch_indices, per_token_losses
                )  # [batch_size] summed losses for each batch where the token is unmasked
                # Count the tokens per batch
                counts = torch.zeros(batch_size, device=device, dtype=per_token_losses.dtype).scatter_add_(
                    0, batch_indices, torch.ones_like(per_token_losses)  # [batch_size] counts for each batch where the token is unmasked
                )
                # Divide sum by count to get mean
                per_token_logps[i, :] = -summed_losses / counts.clamp(min=1.0)  # clamp to avoid division by zero

        # Handle 2D case (different eval_time_steps for each batch)
        elif eval_time_steps.dim() == 2:
            if eval_time_steps.shape[0] != batch_size:
                raise ValueError(f'eval_time_steps first dimension ({eval_time_steps.shape[0]}) must match batch_size ({batch_size})')
            
            num_steps = eval_time_steps.shape[1]
            per_token_logps = torch.zeros(num_steps, batch_size, device=device)  # [num_steps, batch_size]
            if get_all_tokens_logps:
                num_tokens_per_diffusion_step = self.args.max_completion_length // self.args.diffusion_steps
                all_tokens_logps = torch.zeros(num_steps, batch_size, num_tokens_per_diffusion_step, self.vocab_size, device=device)  # [num_steps, batch_size, num_tokens_per_diffusion_step, vocab_size]

            # Process each batch item separately
            for b in range(batch_size):
                batch_eval_time_steps = eval_time_steps[b, :]  # [num_steps] for this batch
                batch_trajectory = trajectory_ids[:, b, :]  # [diffusion_steps, seq_len] for this batch
                batch_final_state = final_state[b, :]  # [seq_len] for this batch

                for i in range(num_steps):
                    step_idx = batch_eval_time_steps[i].item()
                    curr_state = batch_trajectory[step_idx, :].unsqueeze(0)  # [1, seq_len]
                    next_state = batch_trajectory[step_idx + 1, :].unsqueeze(0)  # [1, seq_len]

                    if self.args.pred_state == 'next':
                        positions = curr_state != next_state  # [1, seq_len]
                        targets = next_state[positions]  # [positions.sum()]
                    elif self.args.pred_state == 'final':
                        positions = curr_state != batch_final_state.unsqueeze(0)  # [1, seq_len]
                        targets = batch_final_state.unsqueeze(0)[positions]  # [positions.sum()]
                    else:
                        raise ValueError(f'Invalid pred_state: {self.args.pred_state}')

                    if positions.sum() == 0:
                        # No positions to update, skip or set to zero
                        per_token_logps[i, b] = 0.0
                        continue

                    pred_logits_all = model(curr_state).logits  # [1, seq_len, vocab_size]
                    pred_logits = pred_logits_all[positions]  # [positions.sum(), vocab_size]

                    if get_all_tokens_logps:
                        pred_probs = torch.softmax(pred_logits, dim=1) # [positions.sum(), vocab_size] # predicted probabilities
                        eps = 1e-10
                        pred_logps = torch.log(pred_probs + eps)
                        all_tokens_logps[i, b, :, :] = pred_logps

                    # Compute per-token cross-entropy losses
                    per_token_losses = F.cross_entropy(pred_logits, targets, reduction="none")  # [positions.sum()]

                    # Compute summed loss for this batch item at this time step
                    if self.args.logps_aggregation_mode == 'sum':
                        per_step_losses = per_token_losses.sum()
                    elif self.args.logps_aggregation_mode == 'mean':
                        per_step_losses = per_token_losses.mean()
                    else:
                        raise ValueError(f'Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}')
                    per_token_logps[i, b] = -per_step_losses

        else:
            raise ValueError(f'eval_time_steps must be 1D or 2D, got {eval_time_steps.dim()}D')

        torch.cuda.empty_cache()
        if get_all_tokens_logps:
            return per_token_logps, all_tokens_logps  # [num_steps, batch_size], [num_steps, batch_size, num_tokens_per_diffusion_step, vocab_size]
        else:
            return per_token_logps  # [num_steps, batch_size]


    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self._step % (self.args.logps_eval_num_steps * self.args.num_gradient_steps) == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._cached_inputs = inputs
            else:
                inputs = self._cached_inputs
            self._step += 1
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        print('inputs keys: ', list(inputs[0].keys()), flush=True)
        print('inputs list length: ', len(inputs), flush=True)

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Configuration for the diffusion generation
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
            # Process in batches
            for i in range(0, prompt_ids.size(0), generation_batch_size):
                end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                batch_prompt_ids = prompt_ids[i:end_idx]
                batch_prompt_mask = prompt_mask[i:end_idx]
                # WARNING: Attention masks are not currently used during generation.
                # This works fine as we set num_generations == per_device_train_batch_size (no padding tokens created) in our config, but may cause
                # unintended attention to padding tokens when num_generations is smaller.
                # As currently we find Llada's modeling file does not handle attention mask. We will address this in future update soon.

                # Modify the generate function to return the whole trajectory instead of just the completion tokens
                batch_trajectory_ids, batch_logps, batch_unmasked_prob_distributions, batch_greedy_completions, batch_last_non_eos_steps = self.generate(
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
                )  # (diffusion_steps+1, generation_batch_size, seq_len)

                
                # Permute the trajectory to (generation_batch_size, diffusion_steps+1, seq_len) and add to the list
                trajectory_ids_all.append(batch_trajectory_ids.permute(1, 0, 2))
                logps_all.append(batch_logps.permute(1, 0, 2))  # (generation_batch_size, diffusion_steps+1, num_tokens)
                unmasked_prob_distributions_all.append(batch_unmasked_prob_distributions.permute(1, 0, 2, 3))  # (generation_batch_size, diffusion_steps+1, num_tokens, vocab_size)
                greedy_completions_all.append(batch_greedy_completions.permute(1, 0, 2))  # (generation_batch_size, diffusion_steps, seq_len)
                last_non_eos_steps_all.append(batch_last_non_eos_steps) # (generation_batch_size,)
                del batch_prompt_ids, batch_prompt_mask, batch_trajectory_ids, batch_logps, batch_unmasked_prob_distributions
                torch.cuda.empty_cache()

            # (generation_batch_size, diffusion_steps+1, seq_len) -> (num_batches * generation_batch_size, diffusion_steps+1, seq_len) -> (diffusion_steps+1, per_device_train_batch_size, seq_len)
            trajectory_ids = torch.cat(trajectory_ids_all, dim=0).permute(1, 0, 2) # (diffusion_steps+1, per_device_train_batch_size, seq_len)

            logps = torch.cat(logps_all, dim=0) # (per_device_train_batch_size, diffusion_steps+1, num_tokens)
            unmasked_prob_distributions = torch.cat(unmasked_prob_distributions_all, dim=0)  # (per_device_train_batch_size, diffusion_steps+1, num_tokens, vocab_size)
            greedy_completions = torch.cat(greedy_completions_all, dim=0) # (per_device_train_batch_size, diffusion_steps+1, seq_len)
            last_non_eos_steps = torch.cat(last_non_eos_steps_all, dim=0) # (per_device_train_batch_size,)
        
        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(-1)
        prompt_ids = trajectory_ids[:, :, :prompt_length]
        response_trajectory_ids = trajectory_ids[:, :, prompt_length:]

        self.vocab_size = unmasked_prob_distributions.size(-1)

        # completion_ids = response_trajectory_ids[-1, :, :]  # final state of the trajectory [batch_size, completion_length]

    
        completion_ids_final = response_trajectory_ids[-1, :, :] # [batch_size, completion_length]


        # Mask everything after the first EOS token
        is_eos = completion_ids_final == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(-1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(-1), device=device).expand(is_eos.size(0), -1) # (batch_size, completion_length)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        logits_to_keep = completion_ids_final.size(
            1
        )  # we only need to compute the logits for the completion tokens


        

        def _get_high_entropy_time_steps(unmasked_prob_distributions):
            """
            Select time steps with highest entropy from probability distributions.
            
            Args:
                unmasked_prob_distributions: (batch_size, diffusion_steps+1, num_tokens, vocab_size)
                    Probability distributions for unmasked tokens.
            
            Returns:
                eval_time_steps: (batch_size, logps_eval_num_steps)
                    Indices of time steps with highest entropy.
            """
            EPS = 1e-8  # Small epsilon to avoid log(0)
            batch_size = unmasked_prob_distributions.shape[0]
            eval_time_steps = torch.zeros(
                (batch_size, self.args.logps_eval_num_steps),
                device=device,
                dtype=torch.long
            )

            for batch_idx in range(batch_size):
                prob_distributions = unmasked_prob_distributions[batch_idx]  # (diffusion_steps, num_tokens, vocab_size)

                # Compute entropy in the vocab_size dimension
                entropy = -torch.sum(
                    prob_distributions * torch.log(prob_distributions + EPS),
                    dim=-1
                )  # (diffusion_steps, num_tokens)

                # Average entropy over the num_tokens dimension
                entropy = entropy.mean(dim=-1)  # (diffusion_steps,)

                # Select top-k time steps with highest entropy
                top_k_indices = torch.topk(
                    entropy,
                    k=self.args.logps_eval_num_steps,
                    dim=0
                ).indices  # (logps_eval_num_steps,)
                eval_time_steps[batch_idx] = top_k_indices

            return eval_time_steps


        if self.args.terminate_at_last_non_eos:

            print(f'last_non_eos_steps (batch_size,) = ({last_non_eos_steps})', flush=True)
            print(f'terminating at last non-EOS steps', flush=True)

            if self.args.logps_eval_mode == 'unbiased':
                eval_time_steps = torch.zeros((last_non_eos_steps.size(0), self.args.logps_eval_num_steps), device=device, dtype=torch.long) # (batch_size, logps_eval_num_steps)
                for b in range(last_non_eos_steps.size(0)):
                    if self.args.logps_eval_time_steps_mode == 'random':
                        eval_time_steps_this_batch = torch.randint(0, last_non_eos_steps[b] + 1, (self.args.logps_eval_num_steps,)).to(device)
                        eval_time_steps[b] = eval_time_steps_this_batch
                    elif self.args.logps_eval_time_steps_mode == 'uniform':
                        eval_time_steps_this_batch = torch.linspace(0, last_non_eos_steps[b], self.args.logps_eval_num_steps).long().to(device)
                        eval_time_steps[b] = eval_time_steps_this_batch
                    elif self.args.logps_eval_time_steps_mode == 'high_entropy':
                        unmasked_prob_distributions_this_batch = unmasked_prob_distributions[b, :last_non_eos_steps[b] + 1]
                        eval_time_steps_this_batch = _get_high_entropy_time_steps(unmasked_prob_distributions_this_batch.unsqueeze(0))
                        eval_time_steps[b] = eval_time_steps_this_batch.squeeze(0) # (logps_eval_num_steps,)
                    else:
                        ValueError(f'Invalid logps_eval_time_steps_mode: {self.args.logps_eval_time_steps_mode}')
                print(f'eval_time_steps (batch_size, logps_eval_num_steps) = ({eval_time_steps})', flush=True)


        else:
            if self.args.logps_eval_mode == 'unbiased':
                if self.args.logps_eval_time_steps_mode == 'random':
                    eval_time_steps_1d = torch.randint(0, self.args.diffusion_steps-1, (self.args.logps_eval_num_steps-1,)).to(device)
                    # Add the final time step to eval_time_steps
                    eval_time_steps_1d = torch.cat([eval_time_steps_1d, torch.full((1,), self.args.diffusion_steps-1, device=device)])
                    batch_size = unmasked_prob_distributions.size(0)
                    eval_time_steps = eval_time_steps_1d.unsqueeze(0).repeat(batch_size, 1)
                elif self.args.logps_eval_time_steps_mode == 'uniform':
                    eval_time_steps_1d = torch.linspace(0, self.args.diffusion_steps-2, self.args.logps_eval_num_steps-1).long().to(device)
                    # Add the final time step to eval_time_steps
                    eval_time_steps_1d = torch.cat([eval_time_steps_1d, torch.full((1,), self.args.diffusion_steps-1, device=device)])
                    batch_size = unmasked_prob_distributions.size(0)
                    eval_time_steps = eval_time_steps_1d.unsqueeze(0).repeat(batch_size, 1)
                elif self.args.logps_eval_time_steps_mode == 'high_entropy':
                    eval_time_steps = _get_high_entropy_time_steps(unmasked_prob_distributions)
                else:
                    eval_time_steps = None


        old_per_token_logps = torch.zeros((self.args.logps_eval_num_steps, unmasked_prob_distributions.shape[0]), device=device, dtype=torch.float32)
        for batch_idx in range(unmasked_prob_distributions.shape[0]):
            if self.args.logps_aggregation_mode == 'sum':
                old_per_token_logps[:, batch_idx] = logps[batch_idx, eval_time_steps[batch_idx]].sum(dim=-1) # (self.args.logps_eval_num_steps,)
            elif self.args.logps_aggregation_mode == 'mean':
                old_per_token_logps[:, batch_idx] = logps[batch_idx, eval_time_steps[batch_idx]].mean(dim=-1) # (self.args.logps_eval_num_steps,)
            else:
                raise ValueError(f'Invalid logps_aggregation_mode: {self.args.logps_aggregation_mode}')

        with torch.no_grad():

            if self.beta == 0.0:
                ref_per_token_logps = None
                ref_all_tokens_logps = None
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    if self.args.logps_eval_mode == 'unbiased':
                        ref_per_token_logps, ref_all_tokens_logps = self._get_per_token_logps_unbiased(
                            self.model, trajectory_ids, eval_time_steps=eval_time_steps, get_all_tokens_logps=True)
                    else:
                        ref_per_token_logps = None


        batch_size = unmasked_prob_distributions.shape[0]
        # # To eval_time_steps (batch_size, logps_eval_num_steps) we append -1 to dim=-1 to get choosen_steps (batch_size, logps_eval_num_steps+1)
        # choosen_steps = torch.cat([eval_time_steps, torch.full((batch_size, 1), -1, device=device, dtype=torch.long)], dim=-1)
        # print(f'choosen_steps (per_device_train_batch_size, logps_eval_num_steps+1) = ({choosen_steps})', flush=True)


        completion_ids = greedy_completions[:, :, prompt_length:]  # All completion ids (batch_size, diffusion_steps + 1, seq_len)
        len_completion_ids = completion_ids.size(-1)
        selected_completion_ids = torch.zeros((batch_size, self.args.logps_eval_num_steps + 1, len_completion_ids), device=device, dtype=torch.long)

        for b in range(batch_size):
            for t in range(self.args.logps_eval_num_steps):
                selected_completion_ids[b, t] = completion_ids[b, eval_time_steps[b, t]]

        selected_completion_ids[:, -1] = completion_ids[:, -1] # last step is the final completion id

        print(f'selected_completion_ids (batch_size, logps_eval_num_steps+1, len_completion_ids) = ({selected_completion_ids.shape})', flush=True)


        # Now reshape to obtain the selected completions (batch_size * (logps_eval_num_steps+1), len_completion_ids)
        selected_completion_ids_flattened = selected_completion_ids.reshape(-1, len_completion_ids)
        print(f'selected_completion_ids_flattened (batch_size * (logps_eval_num_steps+1), len_completion_ids) = ({selected_completion_ids_flattened.shape})', flush=True)

        print(f'selected_completion_ids_flattened (batch_size * (logps_eval_num_steps+1), len_completion_ids) = ({selected_completion_ids_flattened.shape})', flush=True)

        # # reshape into (per_device_train_batch_size*(diffusion_steps+1), seq_len)
        # completion_ids = completion_ids.reshape(-1, completion_ids.size(-1)) # (per_device_train_batch_size * (diffusion_steps+1), seq_len)
        
        print('prompts: ', len(prompts), flush=True)
        
        prompts_full = [prompt for prompt in prompts for _ in range(self.args.logps_eval_num_steps+1)] # List with each element of prompts repeated diffusion_steps+1 times

        completions_text = self.processing_class.batch_decode(selected_completion_ids_flattened, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts_full, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
            print('in conversational mode', flush=True)
        else:
            print('in non-conversational mode', flush=True)
            completions = completions_text


        print(f'prompts_full: ', len(prompts_full), flush=True)
        print(f'completions_text: ', len(completions_text), flush=True)
        print(f'completions: ', len(completions), flush=True)


        rewards_per_func = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):

                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs for _ in range(self.args.logps_eval_num_steps+1)] for key in keys}
                if reward_func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                output_reward_func = reward_func(
                    prompts=prompts_full,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                # Convert None values to NaN
                output_reward_func = [
                    reward if reward is not None else torch.nan for reward in output_reward_func
                ]

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts_full[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        print(f'rewards_per_func (batch_size, num_reward_funcs) = ({rewards_per_func.shape})', flush=True)

        # reshape rewards_per_func into (per_device_train_batch_size, diffusion_steps+1, num_reward_funcs)
        rewards_per_func = rewards_per_func.reshape(-1, self.args.logps_eval_num_steps + 1, len(self.reward_funcs))  # only for the selected completions


        final_rewards_per_func = rewards_per_func[:, -1]  # (batch_size, num_reward_funcs)
        final_rewards_per_func_all_devices = gather(final_rewards_per_func)

        print(f'final_rewards_per_func_all_devices (batch_size, num_reward_funcs) = ({final_rewards_per_func_all_devices})', flush=True)


        if self.args.correctness_step_reward_only:
            custom_reward_weights = torch.zeros((self.args.logps_eval_num_steps + 1, len(self.reward_funcs)))
            custom_reward_weights[:, -1] = 1.0 # For all intermediate steps, we use correctness reward only
            custom_reward_weights[-1, :] = 1.0 # For last step, we use all reward functions
            custom_reward_weights = custom_reward_weights.to(device).unsqueeze(0) # (1, logps_eval_num_steps + 1, num_reward_funcs)
            print('only using correctness reward for the intermediate steps', flush=True)
        else:
            custom_reward_weights = self.reward_weights.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, num_reward_funcs)
            

        rewards_per_step = (rewards_per_func * custom_reward_weights).nansum(dim=2)
        
        print(f'rewards_per_step (per_device_train_batch_size, diffusion_steps+1) = ({rewards_per_step.shape})', flush=True)
        
        final_rewards = rewards_per_step[:, -1]  # (per_device_train_batch_size,)
        final_rewards_all_devices = gather(final_rewards)

        print(f'final_rewards_all_devices (batch_size,) = ({final_rewards_all_devices})', flush=True)

        stepwise_advantages = (final_rewards.unsqueeze(1) - rewards_per_step[:, :-1])  # (batch_size, logps_eval_num_steps)



        if self.args.dynamic_lambda1:
            if self.state.global_step % self.args.lambda1_update_steps == 0:
                self.args.lambda1 = self.args.lambda1 / 2       
                print(f'updating lambda1 to {self.args.lambda1}', flush=True)

        if self.args.standard_grpo_returns:
            returns_local_selected = final_rewards.unsqueeze(1)
        else:
            returns_local_selected =  (final_rewards.unsqueeze(1) + self.args.lambda1 * stepwise_advantages)
            if self.args.normalize_returns:
                print('normalizing returns by 1 + lambda1', flush=True)
                returns_local_selected = returns_local_selected / (1 + self.args.lambda1) # (per_device_batch_size, logps_eval_num_steps)
            else:
                print('not normalizing returns', flush=True)


        step_reward_ratios = rewards_per_step[:, :-1] / final_rewards.unsqueeze(1)

        # Gather step_reward_ratios across all devices for plotting
        step_reward_ratios_all_devices = self.accelerator.gather_for_metrics(step_reward_ratios)  # (total_batch_size, logps_eval_num_steps)

        # Gather eval_time_steps across all devices for plotting
        eval_time_steps_all_devices = self.accelerator.gather_for_metrics(eval_time_steps)  # (total_batch_size, logps_eval_num_steps)

        returns = gather(returns_local_selected)

        # After gathering, returns is (net_batch_size, logps_eval_num_steps)

        batch_size, K = returns.shape

        # Group by prompt: [num_prompts, num_generations, K]
        returns_grouped = returns.view(-1, self.num_generations, K)

        print(f'returns_grouped (num_prompts, num_generations, logps_eval_num_steps) = ({returns_grouped.shape})', flush=True)

        mean_grouped_returns = returns_grouped.mean(dim=1, keepdim=True)  # [num_prompts, 1, K]  mean_returns
        print(f'using mean of returns as baseline: mean_grouped_returns = ({mean_grouped_returns})', flush=True)

        # Advantages per step
        advantages_per_step = returns_grouped - mean_grouped_returns      # [num_prompts, num_generations, K]
        advantages_per_step = advantages_per_step.view(batch_size, K)    # [batch_size, K]

        print(f'advantages_per_step (batch_size, logps_eval_num_steps) = ({advantages_per_step.shape})', flush=True)

        # Slice to local process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages_per_step = advantages_per_step[process_slice]


        if self.args.standard_grpo_returns:
            # Add stepwise advantages to advantages_per_step
            advantages_per_step = advantages_per_step + self.args.lambda1 * stepwise_advantages
            if self.args.normalize_returns:
                print('normalizing advantages by (1 + lambda1)', flush=True)
                advantages_per_step = advantages_per_step / (1 + self.args.lambda1) # (batch_size, logps_eval_num_steps)
            else:
                print('not normalizing advantages', flush=True)

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        # self._metrics[mode]["zero_std_ratio"].append(zero_std_ratio)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(final_rewards_per_func_all_devices[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(final_rewards_all_devices.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log per-step reward ratios (rewards_per_step / final_rewards) across all devices
        for step_idx in range(step_reward_ratios_all_devices.shape[1]):
            self._metrics["eval"][f"ablation/step_rewards_ratio_{step_idx}"].append(
                step_reward_ratios_all_devices[:, step_idx].mean().item()
            )

        # Log per-step eval time steps across all devices
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

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(final_rewards_all_devices),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": final_rewards_all_devices.tolist(),
                    }
                    df = pd.DataFrame(table)
                    # Log completions at the correct global step so W&B charts
                    # line up with the trainer's resumed global_step.
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

    def _get_train_sampler(self, dataset):
        """
        Override the parent method to handle the dataset parameter correctly.
        This fixes the TypeError where the method was being called with 2 arguments
        but only expected 1.
        """
        return super()._get_train_sampler()
    
