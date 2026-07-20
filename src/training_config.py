import argparse
import json
import os
import sys
import yaml
from typing import Any, Dict


def _build_parser():
    """Return arguments"""

    parser = argparse.ArgumentParser(description="Evaluation pipeline")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    parser.add_argument("--dataset", type=str, default="wmt25")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_chars", type=int, default=355)

    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./grpo_harder_translate_lora")

    # Unsloth/LoRA
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)


    parser.add_argument("--fast_inference", action="store_true", default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_lora_rank", type=int, default=32)

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # GRPO
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # Reward models
    parser.add_argument("--reward_device", type=str, default="cuda:0")
    parser.add_argument("--td_estimator", type=str, default="sentinel")
    parser.add_argument("--comet_translation_model", type=str, default="average")
    parser.add_argument("--sentinel_model", type=str, default="Prosho/sentinel-src-25")
    parser.add_argument("--sentinel_batch_size", type=int, default=8)

    parser.add_argument("--embedder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cola_model", type=str, default="textattack/roberta-base-CoLA")

    parser.add_argument("--sim_floor", type=float, default=0.78)
    parser.add_argument("--diff_clip", type=float, default=2.0)

    # Reward weights
    parser.add_argument("--w_td_diff_abs", type=float, default=1.0)
    parser.add_argument("--w_td_diff_delta", type=float, default=1.0)
    parser.add_argument("--w_format_strict", type=float, default=0.4)
    parser.add_argument("--w_format_soft", type=float, default=0.1)
    parser.add_argument("--w_format_tag_pen", type=float, default=0.1)
    parser.add_argument("--w_len", type=float, default=0.4)
    parser.add_argument("--w_single_pen", type=float, default=0.05)
    parser.add_argument("--w_rep_pen", type=float, default=0.6)
    parser.add_argument("--w_illegal_chars", type=float, default=0.2)
    parser.add_argument("--w_semantic_sim", type=float, default=0.4)
    parser.add_argument("--w_grammatical_cola", type=float, default=0.6)
    parser.add_argument("--w_grammatical_ltp", type=float, default=0.6)
    
    # W&B
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="mt-breaker")
    parser.add_argument("--wandb_run_name", type=str, default="grpo-train-rollouts")
    parser.add_argument("--wandb_flush_every", type=int, default=10)
    parser.add_argument("--wandb_samples_per_flush", type=int, default=16)

    return parser

def parse_args():
    parser = _build_parser()
    args = parser.parse_args()
    return args

def load_config_file(path: str) -> Dict[str, Any]:
    """Parse a YAML config file"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    
def _cli_provided_keys():
    """
    Return set of argument explicitly passed on the command line.
    """
    parser = _build_parser()
    provided = set()
    for action in parser._actions:
        if action.dest in ("help",):
            continue
        for opt in action.option_strings:
            if opt in sys.argv:
                provided.add(action.dest)
                break
    return provided


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration. Priority:
    1. CLI flags
    2. config file values
    3. argparse defaults
    Returns a simple dict.
    """
    cfg = vars(args).copy()

    if args.config:
        file_cfg = load_config_file(args.config)
        cli_keys = _cli_provided_keys()

        for key, value in file_cfg.items():
            # File value over argparse default, but not over explicit CLI flag.
            if key not in cli_keys and value is not None:
                cfg[key] = value

    return cfg
