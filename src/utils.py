
import math
import re
import torch
import random
import os
from torch.utils.data import Subset

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import List, Dict, Any
import wandb
from transformers import TrainerCallback

from dataset import Wmt25Dataset, EnglishWmtSentencesDataset

def set_seed(seed: int = 42):
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def load_dataset(dataset_name:str,
                 tokenizer, 
                 n_samples=-1,
                 min_tokens=None, 
                 max_tokens=None, 
                 seed=42,
                 apply_chat_template=True,
                 unique_original_sentences=False):
    """Load dataset, randomly subsamples it and filters it"""
    
    # load dataset
    if dataset_name == "wmt25":
        dataset = Wmt25Dataset(tokenizer=tokenizer, apply_chat_template=apply_chat_template)
    elif dataset_name in {"all_wmt"}:
        dataset = EnglishWmtSentencesDataset(tokenizer=tokenizer, apply_chat_template=apply_chat_template)
    elif dataset_name in {"wmt19", "wmt20", "wmt21", "wmt22", "wmt23", "wmt24"}:
        year = int(dataset_name[-2:])
        dataset = EnglishWmtSentencesDataset(
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            year=year,
        )
    else:
        raise ValueError(f"Please specify a supported dataset to load")

    # filter it based on token length
    indices = []
    for i in range(len(dataset)):
        text = dataset[i]["prompt"]
        tok_len = len(tokenizer.encode(text))

        if min_tokens is not None and tok_len < min_tokens:
            continue
        if max_tokens is not None and tok_len > max_tokens:
            continue

        indices.append(i)
    print(f"Samples left after filtering: {len(indices)}")

    if unique_original_sentences:
        unique_indices = []
        seen_sentences = set()
        for i in indices:
            sentence = " ".join(str(dataset[i]["original_sentence"]).split())
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            unique_indices.append(i)
        indices = unique_indices
        print(f"Unique sentences left after deduplication: {len(indices)}")

    # subsample it
    if n_samples == -1:
        sampled_dataset = Subset(dataset, indices)
    else:
        rng = random.Random(seed)
        sampled = rng.sample(indices, min(n_samples, len(indices)))
        sampled_dataset = Subset(dataset, sampled)
    return sampled_dataset


class RolloutWandbLogger:
    """Buffer rows (dicts) and flush to explained W&B Table."""
    def __init__(self, enabled: bool, samples_per_flush: int = 16, max_buffer: int = 512):
        self.enabled = enabled and (wandb is not None)
        self.samples_per_flush = samples_per_flush
        self.max_buffer = max_buffer
        self._buf: List[Dict[str, Any]] = []

    def add_many(self, rows: List[Dict[str, Any]]) -> None:
        if not self.enabled or not rows:
            return
        self._buf.extend(rows)
        if len(self._buf) > self.max_buffer:
            self._buf = self._buf[-self.max_buffer:]

    def flush(self, step: int) -> None:
        if not self.enabled or len(self._buf) == 0:
            return
        take = min(len(self._buf), self.samples_per_flush)
        rows = self._buf[:take]
        self._buf = self._buf[take:]

        columns = [
            "step",
            "original",
            "generation",
            "td_diff_abs",
            "td_diff_delta",
            "format_strict",
            "format_soft",
            "format_tag_pen",
            "len",
            "single_pen",
            "rep_pen",
            "illegal_chars",
            "semantic_sim",
            "grammatical_ltp",
            "grammatical_cola",
            "total",
            "raw_completion",
        ]
        data = [[r.get(c) for c in columns] for r in rows]
        table = wandb.Table(columns=columns, data=data)
        wandb.log({"train_rollouts": table})


class RawDeltaAccumulator:
    def __init__(self, enabled: bool):
        self.enabled = enabled and (wandb is not None)
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.min = None
        self.max = None

    def add(self, raw_deltas: List[float]) -> None:
        if not self.enabled or not raw_deltas:
            return

        finite_values = []
        for value in raw_deltas:
            if isinstance(value, bool):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                finite_values.append(numeric_value)

        if not finite_values:
            return

        self.sum += sum(finite_values)
        self.count += len(finite_values)

        step_min = min(finite_values)
        step_max = max(finite_values)
        self.min = step_min if self.min is None else min(self.min, step_min)
        self.max = step_max if self.max is None else max(self.max, step_max)

    def pop_summary(self) -> Dict[str, float]:
        if not self.enabled or self.count == 0:
            return {}

        summary = {
            "reward/td_raw_delta_mean": self.sum / self.count,
            "reward/td_raw_delta_min": self.min,
            "reward/td_raw_delta_max": self.max,
            "reward/td_raw_delta_count": float(self.count),
        }
        self.reset()
        return summary

class StepSyncCallback(TrainerCallback):
    def __init__(self, rollout_logger, raw_delta_accumulator=None, flush_every=10):
        self.rollout_logger = rollout_logger
        self.raw_delta_accumulator = raw_delta_accumulator
        self.flush_every = flush_every
        self.current_step = [0]

    def on_step_end(self, args, state, control, **kwargs):
        self.current_step[0] = int(state.global_step)

        trainer = kwargs.get("trainer", None)
        is_main = True
        if trainer is not None and hasattr(trainer, "accelerator"):
            is_main = bool(trainer.accelerator.is_main_process)

        if is_main:
            if self.raw_delta_accumulator is not None:
                raw_delta_summary = self.raw_delta_accumulator.pop_summary()
                if raw_delta_summary:
                    wandb.log(raw_delta_summary)

            if state.global_step > 0 and state.global_step % self.flush_every == 0:
                self.rollout_logger.flush(step=int(state.global_step))


@dataclass
class RewardWeights:
    td_diff_abs: float = 0.0
    td_diff_delta: float = 1.0
    format_strict: float = 0.0
    format_soft: float = 0.0
    format_tag_pen: float = 0.0
    len: float = 0.4
    single_pen: float = 0.0
    rep_pen: float = 0.6
    illegal_chars: float = 0.2
    semantic_sim: float = 0.4
    grammatical_ltp: float = 0.4
    grammatical_cola: float = 0.4

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))

# Embedding similarity
class Embedder:
    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"[Embedder] Loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**batch)
        last = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled

    @torch.inference_mode()
    def cosine_sim(self, a: List[str], b: List[str]) -> List[float]:
        ea = self.encode(a)
        eb = self.encode(b)
        sims = (ea * eb).sum(dim=-1)
        return sims.detach().float().cpu().tolist()

# Grammatical Correctness
class AcceptabilityClassifier:
    def __init__(self, model_name: str, device: str):
        self.device = device
        print(f"[CoLA] Loading {model_name} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

        self.accept_label_id = 1
        if hasattr(self.model.config, "id2label") and isinstance(self.model.config.id2label, dict):
            for k, v in self.model.config.id2label.items():
                vv = str(v).lower()
                if "acceptable" in vv or "grammatical" in vv or vv in ("accept", "ok", "yes"):
                    self.accept_label_id = int(k)
                    break

    @torch.inference_mode()
    def p_acceptable(self, texts: List[str], max_length: int = 256) -> List[float]:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        p = probs[:, self.accept_label_id]
        return p.detach().float().cpu().tolist()

