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

        traj = inputs["traj"]
        eval_steps = inputs["eval_steps"]
        step_idx = self._step % self.args.logps_eval_num_steps

        if self.args.logps_eval == 'unbiased':
            if eval_steps is None:
                raise ValueError('eval_steps cannot be None when logps_eval is unbiased')
            logps, all_logps = self._compute_step_logps(
                model, traj,
                eval_time_steps=eval_steps[:, step_idx].unsqueeze(1),
                get_all_tokens_logps=True,
            )
            logps = logps.squeeze(0)
            all_logps = all_logps.squeeze(0)
        else:
            raise ValueError(f'Invalid logps_eval: {self.args.logps_eval}')

        tokens_per_step = self.args.max_completion_length // self.args.diffusion_steps

        if self.beta != 0.0:
            ref_all_logps = inputs["ref_all_logps"][step_idx]
            ref_logps = inputs["ref_logps"][step_idx]
            if self.args.logps_aggregation == 'sum':
                exact_kl = ((torch.exp(all_logps) * (all_logps - ref_all_logps)).sum(dim=-1)).sum(dim=-1)
            elif self.args.logps_aggregation == 'mean':
                exact_kl = ((torch.exp(all_logps) * (all_logps - ref_all_logps)).mean(dim=-1)).mean(dim=-1)
            else:
                raise ValueError(f'Invalid logps_aggregation: {self.args.logps_aggregation}')
            k3_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
        else:
            print('beta = 0.0, so no kl term is computed', flush=True)

        advantages = inputs["step_advs"][:, step_idx]
        old_logps = inputs["old_logps"][step_idx] if self.args.logps_eval_num_steps > 1 else logps.detach()

        coef_1 = torch.exp(logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon_high)
        loss1 = coef_1 * advantages
        loss2 = coef_2 * advantages
        elem_loss = -torch.min(loss1, loss2)

        if self.beta != 0.0:
            if self.args.use_exact_kl:
                elem_loss = elem_loss + self.beta * exact_kl
            else:
                elem_loss = elem_loss + self.beta * k3_kl
        else:
            print('beta = 0.0, so no kl term is computed', flush=True)

        if self.args.logps_aggregation == 'sum':
            loss = elem_loss.mean() / tokens_per_step
        elif self.args.logps_aggregation == 'mean':
            loss = elem_loss.mean()
        else:
            raise ValueError(f'Invalid logps_aggregation: {self.args.logps_aggregation}')
        print('step: ', self._step, 'loss: ', loss, flush=True)

        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            if self.args.use_exact_kl:
                if self.args.logps_aggregation == 'sum':
                    mean_kl = exact_kl.mean() / tokens_per_step
                elif self.args.logps_aggregation == 'mean':
                    mean_kl = exact_kl.mean()
                else:
                    raise ValueError(f'Invalid logps_aggregation: {self.args.logps_aggregation}')
            else:
                mean_kl = k3_kl.mean()
            if self.args.logps_aggregation == 'sum':
                self._metrics[mode]["exact_kl"].append(self.accelerator.gather_for_metrics(exact_kl.mean() / tokens_per_step).mean().item())
            elif self.args.logps_aggregation == 'mean':
                self._metrics[mode]["exact_kl"].append(self.accelerator.gather_for_metrics(exact_kl.mean()).mean().item())
            else:
                raise ValueError(f'Invalid logps_aggregation: {self.args.logps_aggregation}')
            self._metrics[mode]["k3_estimate_kl"].append(self.accelerator.gather_for_metrics(k3_kl.mean()).mean().item())
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        else:
            print('beta = 0.0, so no kl term is logged', flush=True)

        clip_ratio = (loss1 < loss2).float().mean()
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        return loss


    def _gumbel_sample(self, logits, temperature, dtype):
        """
        Gumbel max sampling for categorical distributions.
        Per arXiv:2409.02908, float64 improves perplexity but reduces generation quality for MDM.
        """
        if temperature == 0.0:
            return logits
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
        """Generation code adopted from LLaDA (https://github.com/ML-GSAI/LLaDA)"""
        with torch.amp.autocast(device_type='cuda', enabled=True):
            bs = prompt.shape[0]
            dtype = model.dtype
            x = torch.full((bs, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
            x[:, :prompt.shape[1]] = prompt.clone()

            prompt_index = x != mask_id

            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            steps_per_block = max(1, steps // num_blocks)

            trajectory = [x.clone()]
            is_eos_decoded = torch.full((bs, steps), False, dtype=torch.bool, device=x.device)

            if return_metadata:
                logps_out = []
                probs_out = []
                greedy_out = []

            for block in range(num_blocks):
                start = prompt.shape[1] + block * block_length
                end = prompt.shape[1] + (block + 1) * block_length
                block_mask = x[:, start:end] == mask_id
                n_transfer = self._transfer_counts(block_mask, steps_per_block)

                for i in range(steps_per_block):
                    torch.cuda.empty_cache()
                    mask_index = x == mask_id
                    t = block * steps_per_block + i

                    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                        with torch.cuda.amp.autocast(enabled=self.args.fp16):
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits

                            noisy_logits = self._gumbel_sample(logits, temperature=temperature, dtype=dtype)
                            x0 = torch.argmax(noisy_logits, dim=-1)

                            if return_metadata:
                                x0_greedy = torch.argmax(logits, dim=-1)
                                x0_greedy[~mask_index] = x[~mask_index]
                                greedy_out.append(x0_greedy.clone())

                            del noisy_logits

                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(dtype), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)

                            x0_p[:, end:] = -np.inf
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, x0_p, -np.inf)

                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(bs):
                                n = n_transfer[j, i].item()
                                if n > 0:
                                    _, sel = torch.topk(confidence[j], k=n)
                                    transfer_index[j, sel] = True

                            x[transfer_index] = x0[transfer_index]
                            decoded = x0[transfer_index].reshape(bs, -1)
                            is_eos_decoded[:, t] = (decoded == self.processing_class.eos_token_id).all(dim=1)
                            del confidence

                            trajectory.append(x.clone())
                            if return_metadata:
                                probs = p.to(dtype) if remasking == "low_confidence" else F.softmax(logits.to(dtype), dim=-1)

                                batch_probs, batch_logps = [], []
                                for j in range(bs):
                                    unmask_idx = torch.where(transfer_index[j])[0]
                                    unmask_toks = x0[j, unmask_idx]
                                    tok_probs_dist = probs[j, unmask_idx]
                                    tok_probs = tok_probs_dist.gather(dim=1, index=unmask_toks.unsqueeze(1)).squeeze(1)
                                    batch_probs.append(tok_probs_dist)
                                    batch_logps.append(torch.log(tok_probs))

                                del x0
                                probs_out.append(torch.stack(batch_probs, dim=0))
                                logps_out.append(torch.stack(batch_logps, dim=0))

            trajectory = torch.stack(trajectory, dim=0)
            greedy_out.append(trajectory[-1])

            last_non_eos = steps - 1 - is_eos_decoded.flip(dims=[1]).int().argmin(dim=1)
            print('last_non_eos: ', last_non_eos, flush=True)

            if return_metadata:
                logps_out = torch.stack(logps_out, dim=0)
                probs_out = torch.stack(probs_out, dim=0)
                greedy_out = torch.stack(greedy_out, dim=0)
                return trajectory, logps_out, probs_out, greedy_out, last_non_eos
            else:
                return trajectory


    def forward_process(self, batch, prompt_index, mask_id, seed=None):
        set_seed(seed)
        b, l = batch.shape
        t_p = torch.ones(b, device=batch.device) * self.args.p_mask_prompt

        random_matrix = torch.rand((b, l), device=batch.device)
        is_mask_prompt = prompt_index & (random_matrix < t_p.unsqueeze(1))
        is_mask_completion = ~prompt_index
        is_mask = is_mask_prompt | is_mask_completion

        noisy_batch = torch.where(is_mask, mask_id, batch)
        p_mask = torch.where(
            prompt_index,
            t_p.unsqueeze(1),
            torch.ones_like(t_p).unsqueeze(1),
        )
        return noisy_batch, p_mask

    def _get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        if cfg_scale > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = mask_id
            batch = torch.cat([batch, un_batch])

        logits = model(batch).logits
        if cfg_scale > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits

    def _transfer_counts(self, mask_index, steps):
        """Precompute the number of tokens to unmask at each diffusion step."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        n_transfer = base.expand(-1, steps).clone()
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            n_transfer[indices.unsqueeze(0) < remainder] += 1
        return n_transfer.to(torch.int64)

    def _compute_step_logps(self, model, traj, eval_time_steps, get_all_tokens_logps=False):
        """
        Compute per-step log probabilities over trajectory states.

        Args:
            eval_time_steps: 1D [num_steps] for shared steps across the batch,
                             or 2D [batch_size, num_steps] for per-sample steps.
        """
        print('Unbiased GRPO: _compute_step_logps', flush=True)

        _, bs, seq_len = traj.size()
        device = traj.device
        final_state = traj[-1]

        if not isinstance(eval_time_steps, torch.Tensor):
            eval_time_steps = torch.tensor(eval_time_steps, device=device, dtype=torch.long)

        if eval_time_steps.dim() == 1:
            num_steps = len(eval_time_steps)
            step_logps = torch.zeros(num_steps, bs, device=device)

            for i in range(num_steps):
                curr = traj[eval_time_steps[i]]
                nxt = traj[eval_time_steps[i] + 1]

                if self.args.pred_state == 'next':
                    positions = curr != nxt
                    targets = nxt[positions]
                elif self.args.pred_state == 'final':
                    positions = curr != final_state
                    targets = final_state[positions]
                else:
                    raise ValueError(f'Invalid pred_state: {self.args.pred_state}')

                logits = model(curr).logits[positions]
                tok_losses = F.cross_entropy(logits, targets, reduction="none")

                batch_idx_all = torch.arange(bs, device=device).unsqueeze(1).expand(-1, seq_len)
                batch_idx = batch_idx_all[positions]
                sum_losses = torch.zeros(bs, device=device, dtype=tok_losses.dtype).scatter_add_(0, batch_idx, tok_losses)
                counts = torch.zeros(bs, device=device, dtype=tok_losses.dtype).scatter_add_(0, batch_idx, torch.ones_like(tok_losses))
                step_logps[i] = -sum_losses / counts.clamp(min=1.0)

        elif eval_time_steps.dim() == 2:
            if eval_time_steps.shape[0] != bs:
                raise ValueError(f'eval_time_steps first dim ({eval_time_steps.shape[0]}) must match batch_size ({bs})')

            num_steps = eval_time_steps.shape[1]
            step_logps = torch.zeros(num_steps, bs, device=device)
            if get_all_tokens_logps:
                tokens_per_step = self.args.max_completion_length // self.args.diffusion_steps
                all_logps = torch.zeros(num_steps, bs, tokens_per_step, self.vocab_size, device=device)

            for b in range(bs):
                batch_steps = eval_time_steps[b]
                batch_traj = traj[:, b]
                batch_final = final_state[b]

                for i in range(num_steps):
                    t = batch_steps[i].item()
                    curr = batch_traj[t].unsqueeze(0)
                    nxt = batch_traj[t + 1].unsqueeze(0)

                    if self.args.pred_state == 'next':
                        positions = curr != nxt
                        targets = nxt[positions]
                    elif self.args.pred_state == 'final':
                        positions = curr != batch_final.unsqueeze(0)
                        targets = batch_final.unsqueeze(0)[positions]
                    else:
                        raise ValueError(f'Invalid pred_state: {self.args.pred_state}')

                    if positions.sum() == 0:
                        step_logps[i, b] = 0.0
                        continue

                    logits = model(curr).logits[positions]

                    if get_all_tokens_logps:
                        probs = torch.softmax(logits, dim=1)
                        all_logps[i, b] = torch.log(probs + 1e-10)

                    tok_losses = F.cross_entropy(logits, targets, reduction="none")
                    if self.args.logps_aggregation == 'sum':
                        step_logps[i, b] = -tok_losses.sum()
                    elif self.args.logps_aggregation == 'mean':
                        step_logps[i, b] = -tok_losses.mean()
                    else:
                        raise ValueError(f'Invalid logps_aggregation: {self.args.logps_aggregation}')

        else:
            raise ValueError(f'eval_time_steps must be 1D or 2D, got {eval_time_steps.dim()}D')

        torch.cuda.empty_cache()
        if get_all_tokens_logps:
            return step_logps, all_logps
        return step_logps


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


    def _high_entropy_steps(self, unmask_probs):
        """
        Select time steps with highest average token entropy.

        Args:
            unmask_probs: (batch_size, diffusion_steps, num_tokens, vocab_size)
        Returns:
            eval_steps: (batch_size, logps_eval_num_steps)
        """
        EPS = 1e-8
        bs = unmask_probs.shape[0]
        device = unmask_probs.device
        eval_steps = torch.zeros((bs, self.args.logps_eval_num_steps), device=device, dtype=torch.long)
        for b in range(bs):
            probs = unmask_probs[b]
            entropy = -(probs * torch.log(probs + EPS)).sum(dim=-1).mean(dim=-1)
            eval_steps[b] = torch.topk(entropy, k=self.args.logps_eval_num_steps).indices
        return eval_steps

    def _tokenize_prompts(self, inputs):
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
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        return prompts, prompts_text, prompt_ids, prompt_mask

    def _run_generation(self, model, prompt_ids, prompt_mask):
        # WARNING: Attention masks are not currently used during generation.
        # This works fine as we set num_generations == per_device_train_batch_size (no padding tokens
        # created) in our config, but may cause unintended attention to padding tokens when
        # num_generations is smaller. LLaDA's modeling file does not handle attention mask.
        gen_bs = self.args.generation_batch_size
        traj_list, logps_list, probs_list, greedy_list, eos_list = [], [], [], [], []

        for i in range(0, prompt_ids.size(0), gen_bs):
            end = min(i + gen_bs, prompt_ids.size(0))
            b_prompt = prompt_ids[i:end]
            b_mask = prompt_mask[i:end]

            traj, logps, probs, greedy, last_eos = self.generate(
                model=model,
                prompt=b_prompt,
                steps=self.args.diffusion_steps,
                gen_length=self.args.max_completion_length,
                block_length=self.args.block_length,
                temperature=self.args.temperature or 0.0,
                cfg_scale=self.args.cfg_scale,
                remasking=self.args.remasking,
                mask_id=self.args.mask_id,
                return_metadata=True,
            )

            traj_list.append(traj.permute(1, 0, 2))
            logps_list.append(logps.permute(1, 0, 2))
            probs_list.append(probs.permute(1, 0, 2, 3))
            greedy_list.append(greedy.permute(1, 0, 2))
            eos_list.append(last_eos)
            del b_prompt, b_mask, traj, logps, probs
            torch.cuda.empty_cache()

        traj = torch.cat(traj_list, dim=0).permute(1, 0, 2)
        gen_logps = torch.cat(logps_list, dim=0)
        unmask_probs = torch.cat(probs_list, dim=0)
        greedy_completions = torch.cat(greedy_list, dim=0)
        last_non_eos_steps = torch.cat(eos_list, dim=0)

        return traj, gen_logps, unmask_probs, greedy_completions, last_non_eos_steps

    def _select_eval_steps(self, unmask_probs, last_non_eos_steps):
        device = unmask_probs.device
        bs = unmask_probs.shape[0]
        n_eval = self.args.logps_eval_num_steps
        mode = self.args.logps_eval_time_steps

        if self.args.terminate_at_last_non_eos:
            print(f'last_non_eos_steps (batch_size,) = ({last_non_eos_steps})', flush=True)
            print('terminating at last non-EOS steps', flush=True)
            if self.args.logps_eval == 'unbiased':
                eval_steps = torch.zeros((bs, n_eval), device=device, dtype=torch.long)
                for b in range(bs):
                    cap = last_non_eos_steps[b]
                    if mode == 'random':
                        eval_steps[b] = torch.randint(0, cap + 1, (n_eval,)).to(device)
                    elif mode == 'uniform':
                        eval_steps[b] = torch.linspace(0, cap, n_eval).long().to(device)
                    elif mode == 'high_entropy':
                        eval_steps[b] = self._high_entropy_steps(
                            unmask_probs[b, :cap + 1].unsqueeze(0)
                        ).squeeze(0)
                    else:
                        raise ValueError(f'Invalid logps_eval_time_steps: {mode}')
                print(f'eval_steps (batch_size, logps_eval_num_steps) = ({eval_steps})', flush=True)
        else:
            if self.args.logps_eval == 'unbiased':
                if mode == 'random':
                    base = torch.randint(0, self.args.diffusion_steps - 1, (n_eval - 1,)).to(device)
                    steps_1d = torch.cat([base, torch.full((1,), self.args.diffusion_steps - 1, device=device)])
                    eval_steps = steps_1d.unsqueeze(0).repeat(bs, 1)
                elif mode == 'uniform':
                    base = torch.linspace(0, self.args.diffusion_steps - 2, n_eval - 1).long().to(device)
                    steps_1d = torch.cat([base, torch.full((1,), self.args.diffusion_steps - 1, device=device)])
                    eval_steps = steps_1d.unsqueeze(0).repeat(bs, 1)
                elif mode == 'high_entropy':
                    eval_steps = self._high_entropy_steps(unmask_probs)
                else:
                    eval_steps = None

        return eval_steps

    def _compute_ref_logps(self, traj, eval_steps):
        if self.beta == 0.0:
            return None, None
        with torch.no_grad():
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                if self.args.logps_eval == 'unbiased':
                    return self._compute_step_logps(
                        self.model, traj,
                        eval_time_steps=eval_steps,
                        get_all_tokens_logps=True,
                    )
        return None, None

    def _score_completions(self, prompts, prompts_text, inputs, greedy_completions, prompt_len, eval_steps, device):
        """Decode completions at selected eval steps and compute rewards."""
        bs = greedy_completions.shape[0]
        n_eval = self.args.logps_eval_num_steps
        completion_ids = greedy_completions[:, :, prompt_len:]
        comp_len = completion_ids.size(-1)

        sel_comp_ids = torch.zeros((bs, n_eval + 1, comp_len), device=device, dtype=torch.long)
        for b in range(bs):
            for t in range(n_eval):
                sel_comp_ids[b, t] = completion_ids[b, eval_steps[b, t]]
        sel_comp_ids[:, -1] = completion_ids[:, -1]

        sel_comp_ids_flat = sel_comp_ids.reshape(-1, comp_len)
        print(f'sel_comp_ids_flat (batch_size * (logps_eval_num_steps+1), comp_len) = {sel_comp_ids_flat.shape}', flush=True)

        prompts_rep = [p for p in prompts for _ in range(n_eval + 1)]
        completions_text = self.processing_class.batch_decode(sel_comp_ids_flat, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts_rep, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
            print('in conversational mode', flush=True)
        else:
            print('in non-conversational mode', flush=True)
            completions = completions_text

        print(f'prompts_rep: {len(prompts_rep)}, completions: {len(completions)}', flush=True)

        func_rewards = torch.zeros(len(completions), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                func_name = reward_func.__name__
            with profiling_context(self, func_name):
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [ex[key] for ex in inputs for _ in range(n_eval + 1)] for key in keys}
                if func_name == "coding_reward_func":
                    reward_kwargs["cwd_path"] = os.path.join(self.args.output_dir, "execution_files")
                raw_rewards = reward_func(
                    prompts=prompts_rep,
                    completions=completions,
                    step=self._step,
                    run_name=self.args.output_dir,
                    **reward_kwargs,
                )
                raw_rewards = [r if r is not None else torch.nan for r in raw_rewards]
                func_rewards[:, i] = torch.tensor(raw_rewards, dtype=torch.float32, device=device)

        if torch.isnan(func_rewards).all(dim=1).any():
            nan_idx = torch.isnan(func_rewards).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_kwargs = {key: value[nan_idx] for key, value in reward_kwargs.items()}
            row_kwargs["prompt"] = prompts_rep[nan_idx]
            row_kwargs["completion"] = completions[nan_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        print(f'func_rewards shape: {func_rewards.shape}', flush=True)
        func_rewards = func_rewards.reshape(-1, n_eval + 1, len(self.reward_funcs))
        final_func_rewards = func_rewards[:, -1]
        final_func_rewards_all = gather(final_func_rewards)
        print(f'final_func_rewards_all: {final_func_rewards_all}', flush=True)

        if self.args.correctness_step_reward_only:
            w = torch.zeros((n_eval + 1, len(self.reward_funcs)))
            w[:, -1] = 1.0
            w[-1, :] = 1.0
            w = w.to(device).unsqueeze(0)
            print('only using correctness reward for intermediate steps', flush=True)
        else:
            w = self.reward_weights.to(device).unsqueeze(0).unsqueeze(0)

        step_rewards = (func_rewards * w).nansum(dim=2)
        final_rewards = step_rewards[:, -1]
        final_rewards_all = gather(final_rewards)
        raw_step_advs = final_rewards.unsqueeze(1) - step_rewards[:, :-1]

        print(f'step_rewards shape: {step_rewards.shape}', flush=True)
        print(f'final_rewards_all: {final_rewards_all}', flush=True)

        return (
            func_rewards, final_func_rewards, final_func_rewards_all,
            step_rewards, raw_step_advs, final_rewards, final_rewards_all,
            completions_text,
        )

    def _compute_advantages(self, step_rewards, raw_step_advs, final_rewards, prompts, eval_steps, device):
        """Compute GRPO advantages with optional stepwise return shaping."""
        if self.args.dynamic_lambda1:
            if self.state.global_step % self.args.lambda1_update_steps == 0:
                self.args.lambda1 = self.args.lambda1 / 2
                print(f'updating lambda1 to {self.args.lambda1}', flush=True)

        if self.args.standard_grpo_returns:
            local_returns = final_rewards.unsqueeze(1)
        else:
            local_returns = final_rewards.unsqueeze(1) + self.args.lambda1 * raw_step_advs
            if self.args.normalize_returns:
                print('normalizing returns by 1 + lambda1', flush=True)
                local_returns = local_returns / (1 + self.args.lambda1)
            else:
                print('not normalizing returns', flush=True)

        step_reward_ratios = step_rewards[:, :-1] / final_rewards.unsqueeze(1)
        ratios_all = self.accelerator.gather_for_metrics(step_reward_ratios)
        eval_steps_all = self.accelerator.gather_for_metrics(eval_steps)

        returns = gather(local_returns)
        bs, K = returns.shape
        grouped_returns = returns.view(-1, self.num_generations, K)
        mean_returns = grouped_returns.mean(dim=1, keepdim=True)
        step_advs = (grouped_returns - mean_returns).view(bs, K)

        print(f'grouped_returns shape: {grouped_returns.shape}', flush=True)
        print(f'mean_returns: {mean_returns}', flush=True)
        print(f'step_advs shape: {step_advs.shape}', flush=True)

        proc_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        step_advs = step_advs[proc_slice]

        if self.args.standard_grpo_returns:
            step_advs = step_advs + self.args.lambda1 * raw_step_advs
            if self.args.normalize_returns:
                print('normalizing advantages by (1 + lambda1)', flush=True)
                step_advs = step_advs / (1 + self.args.lambda1)
            else:
                print('not normalizing advantages', flush=True)

        return step_advs, ratios_all, eval_steps_all

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        print('inputs keys: ', list(inputs[0].keys()), flush=True)

        prompts, prompts_text, prompt_ids, prompt_mask = self._tokenize_prompts(inputs)
        prompt_len = prompt_ids.size(-1)

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            traj, gen_logps, unmask_probs, greedy_completions, last_non_eos_steps = \
                self._run_generation(unwrapped_model, prompt_ids, prompt_mask)

        self.vocab_size = unmask_probs.size(-1)

        # Build completion mask up to first EOS
        final_completions = traj[-1, :, prompt_len:]
        is_eos = final_completions == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(-1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(-1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        eval_steps = self._select_eval_steps(unmask_probs, last_non_eos_steps)

        # Compute old log-probs from generation-time distributions
        bs = unmask_probs.shape[0]
        old_logps = torch.zeros((self.args.logps_eval_num_steps, bs), device=device, dtype=torch.float32)
        for b in range(bs):
            if self.args.logps_aggregation == 'sum':
                old_logps[:, b] = gen_logps[b, eval_steps[b]].sum(dim=-1)
            elif self.args.logps_aggregation == 'mean':
                old_logps[:, b] = gen_logps[b, eval_steps[b]].mean(dim=-1)
            else:
                raise ValueError(f'Invalid logps_aggregation: {self.args.logps_aggregation}')

        ref_logps, ref_all_logps = self._compute_ref_logps(traj, eval_steps)

        (func_rewards, final_func_rewards, final_func_rewards_all,
         step_rewards, raw_step_advs, final_rewards, final_rewards_all,
         completions_text) = self._score_completions(
            prompts, prompts_text, inputs, greedy_completions, prompt_len, eval_steps, device
        )

        step_advs, ratios_all, eval_steps_all = self._compute_advantages(
            step_rewards, raw_step_advs, final_rewards, prompts, eval_steps, device
        )

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )
        for i, reward_func in enumerate(self.reward_funcs):
            func_name = reward_func.config._name_or_path.split("/")[-1] if isinstance(reward_func, nn.Module) else reward_func.__name__
            self._metrics[mode][f"rewards/{func_name}"].append(
                torch.nanmean(final_func_rewards_all[:, i]).item()
            )
        self._metrics[mode]["reward"].append(final_rewards_all.mean().item())
        for t in range(ratios_all.shape[1]):
            self._metrics["eval"][f"ablation/step_rewards_ratio_{t}"].append(ratios_all[:, t].mean().item())
        for t in range(eval_steps_all.shape[1]):
            self._metrics["eval"][f"ablation/time_step_{t}"].append(eval_steps_all[:, t].float().mean().item())

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

        completion_ids = greedy_completions[:, :, prompt_len:]
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

    def _get_train_sampler(self, dataset):
        """
        Override the parent method to handle the dataset parameter correctly.
        This fixes the TypeError where the method was being called with 2 arguments
        but only expected 1.
        """
        return super()._get_train_sampler()
