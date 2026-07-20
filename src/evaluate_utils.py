import re
import random
import torch
import json
import gc
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Subset
from dataset import Wmt25Dataset
import os

def clean_generation(text):
    "Given a generated text, it cleans it to get just the modified sentence"
    clean_text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("assistant", "").strip()
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean_text

def write_jsonl(output_path, data) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def write_metric_report(output_path: str, data: List[Dict]) -> None:
    """Write a plain-text per-sample report with key MT difficulty scores."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            td = item.get("translation_difficulty", {})
            sentinel = td.get("sentinel_score")
            comet_nllb = td.get("comet_score_nllb")
            comet_helsinki = td.get("comet_score_helsinki")

            f.write(f"ID: {item.get('id', '')}\n")
            f.write(f"Generation: {item.get('n_gen', '')}\n")
            f.write(f"Original: {item.get('original_sentence', '').strip()}\n")
            f.write(f"Rewrite: {item.get('generated_sentence', '').strip()}\n")
            f.write(f"Sentinel Score: {sentinel}\n")
            f.write(f"Comet_Nllb: {comet_nllb}\n")
            f.write(f"Comet_Helsinki: {comet_helsinki}\n")
            f.write("\n" + "=" * 80 + "\n\n")


def load_model(model_name: str):
    """Helper function to load model and tokenizer"""

    print(f"Loading model {model_name}...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" 

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

def save_results(results, path_to_save):
    """Save results file to json"""
    save_dir = os.path.dirname(path_to_save)
    os.makedirs(save_dir, exist_ok=True)
    print(path_to_save)
    with open(path_to_save, "w", encoding="utf-8") as f:
        for item in results:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True
                ) + "\n"
            )

def free_memory() -> None:
    """Release GPU and CPU memory caches."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
