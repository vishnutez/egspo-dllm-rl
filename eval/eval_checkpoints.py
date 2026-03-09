import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]

        out = generate(
            model,
            input_ids,
            tokenizer,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
        )

        generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
        example_result = [
            {
                "question": questions[j],
                "prompt_input": prompts[j],
                "generations": generated_texts[j],
                "ground_truth": gt_answers[j],
            }
            for j in range(len(gt_answers))
        ]
        all_generations.extend(example_result)
        total_processed += len(generated_texts)
        wall_times.append(time.time() - start_time)

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


def run_evaluation_for_checkpoint(
    model_path,
    checkpoint_path,
    tokenizer,
    dataset,
    args,
    local_rank,
    output_dir=None,
):
    """Run evaluation for a single checkpoint."""
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        local_rank
    )

    if checkpoint_path:
        model = PeftModel.from_pretrained(model, checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            if dist.get_rank() == 0:
                print(f"Rank {local_rank}: Parameters synchronized")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(checkpoint_path):
        model_name = checkpoint_path.split("/")
        model_name = model_name[-2] + "_" + model_name[-1]
    else:
        model_name = "instruct" if "Instruct" in model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    # Use provided output_dir or fall back to args.output_dir
    if output_dir is None:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{args.dataset}_{model_name}_{args.gen_length}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    if dist.get_rank() == 0:
        print(f"Saving generations to {filename}")

    if args.skip_existing:
        # Check on rank 0 and broadcast the decision to all ranks
        file_exists = torch.tensor(int(os.path.exists(filename)), device=local_rank)
        if dist.get_world_size() > 1:
            dist.broadcast(file_exists, src=0)
        if file_exists.item():
            if dist.get_rank() == 0:
                print(f"Skipping (output already exists): {filename}")
            return

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        temperature=args.temperature,
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "model_path": model_path,
                    "checkpoint_path": checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )

    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    init_seed(42)

    # Note: This evaluation script saves only model generations. A separate parser is used later to extract
    # predictions and calculate metrics.

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "math", "countdown", "sudoku", "game24"], default="gsm8k"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_checkpoints", type=int, default=100)
    parser.add_argument("--skip_existing", action="store_true", help="Skip checkpoints whose output file already exists")
    args = parser.parse_args()


    print(f"Local rank: {local_rank}", flush=True)
    print(f"Model path: {args.model_path}", flush=True)
    print(f"Few shot: {args.few_shot}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Suffix: {args.suffix}", flush=True)
    print(f"Checkpoint path: {args.checkpoint_path}", flush=True)
    print(f"Checkpoint dir: {args.checkpoint_dir}", flush=True)
    print(f"Gen length: {args.gen_length}", flush=True)
    print(f"Block length: {args.block_length}", flush=True)
    print(f"Diffusion steps: {args.diffusion_steps}", flush=True)
    print(f"Add reasoning: {args.add_reasoning}", flush=True)
    print(f"Don't save: {args.dont_save}", flush=True)
    print(f"Output dir: {args.output_dir}", flush=True)
    print(f"Don't use box: {args.dont_use_box}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    args.diffusion_steps = args.gen_length // 2
    num_evals = {"gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Add barrier to synchronize processes before dataset loading
    # This prevents race conditions when multiple processes try to access the dataset cache simultaneously
    if dist.get_world_size() > 1:
        dist.barrier()

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill for all models
    )

    # Determine checkpoints to evaluate
    if args.checkpoint_dir is not None:
        # Get all items in the checkpoint directory
        checkpoint_paths = []
        checkpoint_names = []
        if os.path.isdir(args.checkpoint_dir):
            items = sorted(os.listdir(args.checkpoint_dir))[::-1][:args.num_checkpoints]
            if dist.get_rank() == 0:
                print(f"Evaluating {args.num_checkpoints} checkpoints from {args.checkpoint_dir}")
                for item in items:
                    print(f"  - {item}")
            for item in items:
                item_path = os.path.join(args.checkpoint_dir, item)
                # Check if it's a directory (typical for PEFT checkpoints) or a file
                if os.path.isdir(item_path) or os.path.isfile(item_path):
                    checkpoint_paths.append(item_path)
                    checkpoint_names.append(item)  # Store the checkpoint name (e.g., "checkpoint-1000")
        
        if dist.get_rank() == 0:
            print(f"Found {len(checkpoint_paths)} checkpoints in {args.checkpoint_dir}")
            for cp in checkpoint_paths:
                print(f"  - {cp}")
        
        # Loop through all checkpoints
        for checkpoint_path, checkpoint_name in zip(checkpoint_paths, checkpoint_names):
            if dist.get_rank() == 0:
                print(f"\n{'='*80}")
                print(f"Evaluating checkpoint: {checkpoint_path}")
                print(f"{'='*80}\n")
            
            # Create checkpoint-specific output directory using checkpoint name
            checkpoint_output_dir = os.path.join(args.output_dir, checkpoint_name)
            
            run_evaluation_for_checkpoint(
                args.model_path,
                checkpoint_path,
                tokenizer,
                dataset,
                args,
                local_rank,
                output_dir=checkpoint_output_dir,
            )
    else:
        # Original single checkpoint evaluation
        run_evaluation_for_checkpoint(
            args.model_path,
            args.checkpoint_path,
            tokenizer,
            dataset,
            args,
            local_rank,
        )

    cleanup_ddp()
