import torch
import wandb
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig

# Custom imports
from epsa_trainer import EPSATrainer
from epsa_config import EPSAConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    set_random_seed,
    get_math_questions,
)


# # Force offline mode at the OS level before any HF libraries load
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"


def main(grpo_config, model_config):

    # # Initialize wandb if WANDB_ID is set (for resuming)
    # # Only initialize on rank 0 to avoid multiple wandb instances in distributed training

    rank_str = os.environ.get("RANK")
    rank = int(rank_str) if rank_str is not None else 0

    wandb_id = os.environ.get("WANDB_ID", False)

    maybe_resume_from_checkpoint = True if wandb_id else False
    print(f"Maybe resume from checkpoint: {maybe_resume_from_checkpoint}", flush=True)


    if maybe_resume_from_checkpoint and rank == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "huggingface"),
            resume="must",
            id=wandb_id,
        )
        print(f"Initialized wandb with run_id={wandb_id}", flush=True)
    else:
        print(f"Initializing a new wandb run", flush=True)

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
        test_set = get_gsm8k_questions("test")
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            boxed_and_answer_tags_format_reward,
            correctness_reward_func_math,
        ]
        test_set = get_math_questions("test")

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
        test_set = dataset.select(range(len(dataset) - 500, len(dataset)))  # Last 500 for evaluation
    else:
        train_set = dataset

    print('Length of train set:', len(train_set), flush=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print('Loading model... from', f'{grpo_config.model_path}', flush=True)

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        local_files_only=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = EPSATrainer(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )

    trainer.train(resume_from_checkpoint=maybe_resume_from_checkpoint)


if __name__ == "__main__":
    parser = TrlParser((EPSAConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
