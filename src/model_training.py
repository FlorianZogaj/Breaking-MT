from unsloth import FastLanguageModel
import argparse
import os
import re
import math
import random
from typing import List, Dict, Any, Optional
from types import SimpleNamespace
import yaml

import torch
# from sentinel_metric import download_model as sentinel_download_model
# from sentinel_metric import load_from_checkpoint as sentinel_load_from_checkpoint
from datasets import load_dataset, Dataset

from datasets import Dataset as HFDataset


from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainerCallback

from trl import GRPOConfig, GRPOTrainer

import wandb

from utils import (
    RewardWeights,
    RawDeltaAccumulator,
    RolloutWandbLogger,
    StepSyncCallback,
    set_seed
)
from rewards import (
    RewardComputer,
)

from training_config import (
    parse_args,
    build_config
)


if os.path.exists("wandb.key"):
    print("Found wandb.key, logging in...")
    with open("wandb.key") as f:
        wandb.login(key=f.read().strip())
else:
    print("No wandb.key found")

def get_dataset(cfg: Dict[str, Any], tokenizer) -> Dataset:
    from utils import load_dataset
    ds = load_dataset(cfg["dataset"], tokenizer)
    print(f"[Dataset] {len(ds)} examples loaded.")
    return ds

def load_model(cfg: Dict[str, Any]):
    # Expand environment variables in base_model path
    base_model_path = os.path.expandvars(cfg["base_model"])
    print(f"[Model] Loading {base_model_path} ...")
    fp_kwargs = dict(
        model_name=base_model_path,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    if cfg["fast_inference"]:
        fp_kwargs.update(dict(
            fast_inference=True,
            max_lora_rank=cfg["max_lora_rank"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
        ))

    model, tokenizer = FastLanguageModel.from_pretrained(**fp_kwargs)

    target_modules = [m.strip() for m in cfg["target_modules"].split(",") if m.strip()]
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        target_modules=target_modules,
        lora_alpha=cfg["lora_rank"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    return model, tokenizer

def build_reward_computer(cfg: Dict[str, Any], rollout_logger, raw_delta_accumulator):
    reward_device = cfg["reward_device"]
    if reward_device.startswith("cuda") and not torch.cuda.is_available():
        reward_device = "cpu"

    weights = RewardWeights(
        td_diff_abs=cfg["w_td_diff_abs"],
        td_diff_delta=cfg["w_td_diff_delta"],
        format_strict=cfg["w_format_strict"],
        format_soft=cfg["w_format_soft"],
        format_tag_pen=cfg["w_format_tag_pen"],
        len=cfg["w_len"],
        single_pen=cfg["w_single_pen"],
        rep_pen=cfg["w_rep_pen"],
        illegal_chars=cfg["w_illegal_chars"],
        semantic_sim=cfg["w_semantic_sim"],
        grammatical_ltp=cfg["w_grammatical_ltp"],
        grammatical_cola=cfg["w_grammatical_cola"],
    )

    reward_func = RewardComputer(
        weights=weights,
        td_estimator=cfg["td_estimator"],
        comet_translation_model=cfg["comet_translation_model"],
        embedder_name=cfg["embedder_model"],
        cola_name=cfg["cola_model"],
        rollout_logger=rollout_logger,
        raw_delta_accumulator=raw_delta_accumulator,
        device=reward_device
    )
    return reward_func


def main():
    args = parse_args()
    cfg = build_config(args)
    
    set_seed(cfg["seed"])

    # wandb
    if cfg["wandb"]:
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)

    rollout_logger = RolloutWandbLogger(
        enabled=cfg["wandb"],
        samples_per_flush=cfg["wandb_samples_per_flush"],
        max_buffer=512,
    )
    raw_delta_accumulator = RawDeltaAccumulator(enabled=cfg["wandb"])

    # load and build 
    model, tokenizer = load_model(cfg)
    dataset = get_dataset(cfg, tokenizer)
    reward_computer = build_reward_computer(cfg, rollout_logger, raw_delta_accumulator)
    
    def reward_func(completions, original_sentence, **kwargs):
        return reward_computer(
            completions=completions,
            original_sentence=original_sentence,
            **kwargs,
        )
        
    
    print("CONFIG:")
    print(cfg["max_prompt_length"])
    print(cfg["max_completion_length"])
    training_args = GRPOConfig(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        num_generations=cfg["num_generations"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        report_to=(["wandb"] if cfg["wandb"] else ["none"]),
        run_name=(cfg["wandb_run_name"] if cfg["wandb"] else None),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.add_callback(
        StepSyncCallback(
            rollout_logger,
            raw_delta_accumulator=raw_delta_accumulator,
            flush_every=cfg["wandb_flush_every"],
        )
    )

    trainer.train()

    final_raw_delta_summary = raw_delta_accumulator.pop_summary()
    if final_raw_delta_summary:
        wandb.log(final_raw_delta_summary)
    rollout_logger.flush(step=trainer.state.global_step)

    print(f"[Save] Saving adapter to {cfg['output_dir']} ...")
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("[Done]")

if __name__ == "__main__":
    main()
