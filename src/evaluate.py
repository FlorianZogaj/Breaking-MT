from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch

from tqdm import tqdm

from evaluate_utils import (
    load_model,
    clean_generation,
    write_jsonl,
    write_metric_report,
)
from utils import (
    load_dataset, 
    set_seed
)
from transformers import AutoTokenizer

from evaluate_config import (
    parse_args,
    build_config,
    _jsonl_path,
    _txt_report_path,
)

def generate_responses(
    cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate responses from a given model. 
    Returns a list of dicts, each containing: id, original_sentence, prompt, generated_sentence."""

    # load model, dataset
    model, tokenizer = load_model(model_name=cfg["model"])
    dataset = load_dataset(
        dataset_name=cfg["dataset"],
        tokenizer=tokenizer,
        n_samples=cfg["n_samples"],
        apply_chat_template=True,
        unique_original_sentences=cfg.get("unique_original_sentences", False),
    )

    # generate responses
    results = []
    batch_size = cfg["batch_size"]
    for n_gen in range(cfg["num_generation"]):
        for start in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
            batch_items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]

            ids = [item["id"] for item in batch_items]
            originals = [item["original_sentence"] for item in batch_items]
            prompts = [item["prompt"] for item in batch_items]

            
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)
            prompt_width = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_new_tokens"],
                    temperature=cfg["temperature"],
                    do_sample=cfg["do_sample"],
                    top_p=cfg["top_p"],
                )

            for i in range(len(batch_items)):
                gen_tokens = outputs[i, prompt_width:]
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                generated = clean_generation(decoded)
                results.append(
                    {
                        "id": ids[i],
                        "n_gen": n_gen,  
                        "original_sentence": originals[i],
                        "prompt": prompts[i],
                        "generated_sentence": generated,
                    }
                )

    return results

def collect_originals(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If no generation model specified, treat original as generated output.
    """
    print("Evaluating original sentences directly.")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset(
        dataset_name=cfg["dataset"],
        tokenizer=tokenizer,
        n_samples=cfg["n_samples"],
        apply_chat_template=False, 
        unique_original_sentences=cfg.get("unique_original_sentences", False),
    )

    results = []
    for i in range(len(dataset)):
        item = dataset[i]
        results.append(
            {
                "id": item["id"],
                "n_gen": 1,
                "original_sentence": item["original_sentence"],
                "prompt": None,
                "generated_sentence": item["original_sentence"],
            }
        )
    return results

def run_pipeline(cfg: Dict[str, Any]) -> None:
    jsonl_path = _jsonl_path(cfg)
    txt_report_path = _txt_report_path(cfg)

    # generate samples
    if cfg.get("model") is None:
        samples = collect_originals(cfg)
    else:
        samples = generate_responses(cfg)

    # score samples
    samples = score_translation_difficulty(samples)
    samples = score_grammatical_errors(samples)
    samples = score_complexity(samples)    
    samples = score_embeddings(samples, cfg)
    samples = score_judge(samples, cfg)

    write_jsonl(jsonl_path, samples)
    print(f"Saved {len(samples)} scored samples to {jsonl_path}")
    
    write_metric_report(txt_report_path, samples)
    print(f"Saved per-sample metric report to {txt_report_path}")

    # aggregate scores
    summary = aggregate_results(samples, cfg)

    summary_path = jsonl_path.replace("samples.jsonl", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary saved → {summary_path!r}")
    print(json.dumps(summary, indent=2))

from evaluate_scores import (
    score_translation_difficulty,
    score_grammatical_errors,
    score_complexity, 
    score_embeddings,
    score_judge,
    aggregate_results
)

def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg["seed"])
    
    run_pipeline(cfg)


if __name__ == "__main__":
    # NCCL tweaks are harmless on a single GPU and can help on some multi-GPU boxes.
    # NB: do NOT force the "spawn" start method here — the COMET/sentinel scorers use
    # PyTorch-Lightning predict dataloaders that fail to share CUDA tensors under spawn.
    # The default (fork on Linux) is required for evaluation to run.
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    main()
