import argparse
import json
import os
import sys
import yaml
from typing import Any, Dict

def _jsonl_path(cfg: Dict[str, Any]) -> str:
    model_tag = os.path.basename(cfg["model"]).replace("/", "_") if cfg.get("model") else "originals"
    name = f"{model_tag}_{cfg['dataset']}_n{cfg['n_samples']}"
    return os.path.join(cfg["path_to_save"], name, "samples.jsonl")

def _txt_report_path(cfg: Dict[str, Any]) -> str:
    model_tag = os.path.basename(cfg["model"]).replace("/", "_") if cfg.get("model") else "originals"
    name = f"{model_tag}_{cfg['dataset']}_n{cfg['n_samples']}"
    return os.path.join(cfg["path_to_save"], name, "metric_report.txt")


def _build_parser():
    """Return arguments"""

    parser = argparse.ArgumentParser(description="Evaluation pipeline")

    # Config file
    parser.add_argument("--config", default=None, help="Path to YAML config file. CLI flags override its values.",)

    # Models
    parser.add_argument("--model", default=None, help="Path to the generation model. If None, original sentences are used for evaluation.")
    parser.add_argument("--model_judge", default=None, help="Path to the judge LLM")

    # Dataset
    parser.add_argument("--dataset", default="wmt25", choices=["wmt19", "wmt20", "wmt21", "wmt22", "wmt23", "wmt24", "wmt25", "all_wmt"], help="Dataset to evaluate on.")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to draw from the dataset. If = -1 then all samples are considered.")
    parser.add_argument(
        "--unique_original_sentences",
        action="store_true",
        default=False,
        help="Deduplicate by original sentence before sampling.",
    )

    # Generation
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens per sample.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling probability.")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Sample during inference.")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_generation", type=int, default=2, help="How many times to repeat inference")
    parser.add_argument("--aggregation_mode", choices=["best_of", "all"], default="best_of", help="Aggregation mode of the metrics")

    # Paths
    parser.add_argument("--path_to_save", default="./evaluation/", help="Root directory for saving outputs.")

    # Embedding similarity
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Model to use for embedding similarity")
    parser.add_argument("--device", default="cuda", help="Which device to use")
    parser.add_argument("--ngram_top_k", type=int, default=10, help="How many top n-grams to report in the summary.")
    parser.add_argument("--ngram_sizes", default="1,2", help="Comma-separated n-gram sizes to compute, e.g. '1,2,3'.")

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

    sizes = cfg.get("ngram_sizes", "1,2")
    if isinstance(sizes, str):
        cfg["ngram_sizes"] = [int(part.strip()) for part in sizes.split(",") if part.strip()]
    elif isinstance(sizes, (list, tuple)):
        cfg["ngram_sizes"] = [int(part) for part in sizes]
    else:
        cfg["ngram_sizes"] = [1, 2]

    return cfg
