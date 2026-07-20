
import nltk
import torch
import language_tool_python as tlp
import numpy as np
import re 
import json
import math
import spacy
from sacrebleu.metrics import CHRF
import textdescriptives as td
from typing import Dict, List, Any, Optional
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from collections import Counter, defaultdict


from evaluate_utils import (
    free_memory
)
from model.sentinel import SentinelScorer
from model.comet import CometScorer

from evaluate_utils import (
    load_model,
    clean_generation
)

from prompts import (
    JUDGE_PROMPT
)

_SPACY_NLP = None

# ====  Translation difficulty scores
def score_translation_difficulty(
    samples: List[Dict[str, Any]],
    path_madlad: str = None,
    path_helsinki: str = None, 
    path_nllb: str = None,
) -> List[Dict[str, Any]]:
    """Sentinel and Comet scores for generated text"""
    generated = [s["generated_sentence"] for s in samples]

    _add_sentinel_scores(samples, generated)
    free_memory()
    _add_comet_scores(samples, generated, path_madlad, path_helsinki, path_nllb)
    free_memory()

    return samples

def _add_sentinel_scores(samples: List[Dict[str, Any]], texts: List[str],) -> None:
    """Compute sentinel scores and store under samples[i]["translation_difficult"]["sentinel_score"]."""
    print("Computing sentinel scores...")
    scorer = SentinelScorer()
    scores = scorer.assign_score(texts)["scores"]
    for i, s in enumerate(samples):
        s.setdefault("translation_difficulty", {})["sentinel_score"] = float(scores[i])

def _add_comet_scores(samples: List[Dict[str, Any]], texts: List[str], path_madlad: str, path_helsinki: str, path_nllb: str) -> None:
    """Compute Comet scores and store under samples[i]["translation_difficult"]["comet_score"]."""
    print("Computing COMET scores...")
    scorer = CometScorer()
    scores = scorer.assign_all_scores(texts)
    for i, s in enumerate(samples):
        td = s.setdefault("translation_difficulty", {})
        td["comet_score"] = float(scores["average"][i])
        td["comet_score_nllb"] = float(scores["nllb"][i])
        td["comet_score_helsinki"] = float(scores["helsinki"][i])

# ==== Grammatical errors score
def score_grammatical_errors(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute number of grammatical errors using tlp"""
    print("Computing grammatical errors score...")

    tool = tlp.LanguageTool("en-US")
    for s in samples:
        matches = tool.check(s.get("generated_sentence", ""))
        s["grammatical_errors"] = len(matches)
    return samples

# ==== Complexity score
def score_complexity(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    "Compute rix, entropy, avg word lenght, avg sentence length, and total word count."
    print("Computing complexity scores…")

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    stop_set = set(stopwords.words("english"))

    generated = [s["generated_sentence"] for s in samples]
    df = td.extract_metrics(text=generated, lang="en", metrics=["readability", "information_theory"])
    rix = df["rix"].to_numpy()
    entropy = df["entropy"].to_numpy()

    for i, s in enumerate(samples):
        s.setdefault("complexity_score", {}).update(
            {
                "rix": float(rix[i]),
                "entropy": float(entropy[i]),
                "avg_word_length": _avg_word_length(generated[i], stop_set),
                "avg_sentence_length": _avg_sentence_length(generated[i]),
                "total_word_count": _total_word_count(generated[i]),
            }
        )
    return samples

def _avg_word_length(text, stop_set):
    words = [w.lower() for w in word_tokenize(text) if w.lower() not in stop_set]
    return float(np.mean([len(w) for w in words])) if words else 0.0

def _avg_sentence_length(text):
    parts = [p.strip() for p in text.split(".") if p.strip()]
    lengths = [len(p.split()) for p in parts]
    return float(np.mean(lengths)) if lengths else 0.0

def _total_word_count(text):
    return float(len(re.findall(r"\b\w+\b", text))) if text else 0.0

# ==== LLM Judge
def score_judge(samples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score using a judge LLM. Evaluating naturalness, word rarity, syntax complexity, topics."""
    print("Computing judge scores…")

    if not cfg.get("model_judge"):
        print("No judge model specified (model_judge is null); skipping judge scores.")
        return samples

    # load model
    model, tokenizer = load_model(model_name=cfg["model_judge"])
    generated = [s["generated_sentence"] for s in samples]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": JUDGE_PROMPT.format(source_text=text)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for text in generated
    ]

    all_outputs = []
    for start in tqdm(range(0, len(prompts), cfg["batch_size"]), desc="Judge evaluation"):
        batch = prompts[start : start + cfg["batch_size"]]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )

        trimmed = [
            out[len(inp) :]
            for inp, out in zip(inputs.input_ids, outputs)
        ]
        all_outputs.extend(
            tokenizer.batch_decode(trimmed, skip_special_tokens=True)
        )

    for i, raw in enumerate(all_outputs):
        parsed = _parse_judge_output(raw)
        cs = samples[i].setdefault("complexity_score", {})
        ds = samples[i].setdefault("diversity_score", {})

        if not parsed:
            cs.update({"naturalness": None, "word_rarity": None, "syntax_complexity": None})
            ds["topics"] = None
            continue

        cs["naturalness"] = _extract_numeric(parsed, "naturalness")
        cs["word_rarity"] = _extract_numeric(parsed, "word rarity")
        cs["syntax_complexity"] = _extract_numeric(parsed, "syntax complexity")

        topics = parsed.get("topics")
        ds["topics"] = [str(t) for t in topics[:5]] if isinstance(topics, list) else None

    return samples

def _parse_judge_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from a judge model response"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    candidate = re.sub(r",\s*}", "}", m.group(0).strip())
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_numeric(d: Dict[str, Any], key: str):
    v = d.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(\.\d+)?", v)
        return float(m.group(0)) if m else None
    return None


# ==== Embedding score 
def score_embeddings(samples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute embedding score."""
    print("Computing per-sample embeddings…")
    sim_model = SentenceTransformer(cfg["embedding_model"], device=cfg["device"])
    texts = [clean_generation(s.get("generated_sentence", "")) for s in samples]
    embeddings = sim_model.encode(
        texts, convert_to_tensor=True, show_progress_bar=True, batch_size=128
    )
    for i, s in enumerate(samples):
        s["embedding"] = embeddings[i].cpu().tolist()
    return samples

# ==== Aggregation

def aggregate_results(samples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full aggregation pipeline:
      1. Select samples according to cfg["aggregation_mode"].
      2. Aggregate per-sample scalar metrics (phase 1).
      3. Aggregate corpus-level embedding similarities (phase 2).
    Returns the merged summary dict.
    """
    selected = select_samples(samples, cfg)

    agg = aggregate_metrics(selected)
    corpus_agg = compute_corpus_metrics(selected, cfg)
    agg.update(corpus_agg)

    return agg

def select_samples(data: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Select samples to aggregate on:
      - "all":     keep every sample as-is.
      - "best_of": per id, keep the generation with the highest sentinel_score + comet_score.
    """
    mode = cfg.get("aggregation_mode", "all")
    if mode == "all":
        return data
    elif mode == "best_of":
        groups = defaultdict(list)
        for r in data:
            groups[r["id"]].append(r)
        return [
            max(rows, key=lambda r: (
                (_translation_metric(r, "sentinel_score") or 0.0)
                + (_translation_metric(r, "comet_score") or 0.0)
            ))
            for rows in groups.values()
        ]
    else:
        raise ValueError(f"Unknown aggregation_mode: {mode!r}. Use 'all' or 'best_of'.")

def _safe_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
        return v if math.isfinite(v) else None
    return None


def aggregate_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate previous metrics across samples
    """
    print("Aggregating per-sample metrics…")
    agg = {}

    # Translation difficulty 
    agg["translation_difficulty"] = {
        m: float(np.mean(v)) if (v := _collect_translation_metric(samples, m)) else None
        for m in ("sentinel_score", "comet_score", "comet_score_nllb", "comet_score_helsinki")
    }

    # Complexity
    agg["complexity_score"] = {
        m: float(np.mean(v)) if (v := _collect(samples, "complexity_score", m)) else None
        for m in (
            "rix", "entropy", "avg_word_length", "avg_sentence_length", "total_word_count",
            "naturalness", "word_rarity", "syntax_complexity",
        )
    }

    # Grammatical errors
    ge_vals = _collect(samples, "grammatical_errors")
    agg["grammatical_errors"] = float(np.mean(ge_vals)) if ge_vals else None

    # Diversity — unique topics
    all_topics: set = set()
    for s in samples:
        topics = s.get("diversity_score", {}).get("topics") or []
        all_topics.update(t.strip() for t in topics if isinstance(t, str) and t.strip())
    agg["unique_topics_count"] = len(all_topics)
    agg["num_rows"] = len(samples)

    return agg


def compute_corpus_metrics(samples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute corpus-level metrics
    """
    print("Computing corpus-level embedding similarities…")
    agg = {}
    generated_texts = [s.get("generated_sentence", "") for s in samples]

    try:
        agg["embedding_similarity_generated_pairwise"] = corpus_embedding(generated_texts, cfg)
    except Exception as e:
        agg["embedding_similarity_generated_pairwise"] = str(e)

    try:
        agg["self_chrf_generated_pairwise"] = corpus_self_chrf(generated_texts)
    except Exception as e:
        agg["self_chrf_generated_pairwise"] = str(e)

    try:
        agg["ngram_frequency"] = corpus_ngram_frequency(samples, cfg)
    except Exception as e:
        agg["ngram_frequency"] = {"error": str(e)}

    return agg


def corpus_embedding(texts: List[str], cfg: Dict[str, Any]) -> Optional[float]:
    clean = [clean_generation(t) for t in texts if isinstance(t, str) and clean_generation(t)]
    if len(clean) < 2:
        return None
    sim_model = SentenceTransformer(cfg["embedding_model"], device=cfg["device"])
    
    clean_texts = [clean_generation(t) for t in texts]
    emb = sim_model.encode(
        clean_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    sim = util.pytorch_cos_sim(emb, emb)
    n = sim.size(0)
    idx = torch.triu_indices(n, n, offset=1, device=sim.device)
    return float(sim[idx[0], idx[1]].mean().item())


def corpus_self_chrf(texts: List[str]) -> Optional[float]:
    """
    Mean pairwise sentence-level chrF across generated outputs.
    Higher values indicate more similar generations and potentially more collapse.
    """
    clean_texts = [
        clean_generation(text)
        for text in texts
        if isinstance(text, str) and clean_generation(text)
    ]
    if len(clean_texts) < 2:
        return None

    metric = CHRF()
    total = 0.0
    count = 0

    for i in range(len(clean_texts)):
        for j in range(i + 1, len(clean_texts)):
            total += float(metric.sentence_score(clean_texts[i], [clean_texts[j]]).score)
            count += 1

    return total / count if count else None


def _collect(samples: List[Dict[str, Any]], *keys: str) -> List[float]:
    out = []
    for s in samples:
        d: Any = s
        for k in keys:
            if not isinstance(d, dict):
                d = None
                break
            d = d.get(k)
        v = _safe_float(d)
        if v is not None:
            out.append(v)
    return out


def corpus_ngram_frequency(samples: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    nlp = _load_spacy_model()
    original_texts = [s.get("original_sentence", "") for s in samples]
    generated_texts = [s.get("generated_sentence", "") for s in samples]
    ngram_sizes = cfg.get("ngram_sizes") or [1, 2]
    top_k = int(cfg.get("ngram_top_k", 10))

    summary = {
        "top_k": top_k,
        "ngram_sizes": ngram_sizes,
        "original": {},
        "generated": {},
    }

    for n in ngram_sizes:
        summary["original"][f"{n}gram"] = _get_top_phrases_lemmatized(original_texts, nlp, n=n, top_k=top_k)
        summary["generated"][f"{n}gram"] = _get_top_phrases_lemmatized(generated_texts, nlp, n=n, top_k=top_k)

    return summary


def _load_spacy_model():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is required for n-gram frequency analysis. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from exc
    return _SPACY_NLP


def _get_top_phrases_lemmatized(
    sentences: List[str],
    nlp,
    n: int = 1,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    counter: Counter = Counter()

    for doc in nlp.pipe(sentences, batch_size=256):
        lemmas = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop
        ]

        if n == 1:
            counter.update(lemmas)
        elif len(lemmas) >= n:
            counter.update(ngrams(lemmas, n))

    results = []
    for phrase, count in counter.most_common(top_k):
        if isinstance(phrase, tuple):
            phrase_text = " ".join(phrase)
        else:
            phrase_text = phrase
        results.append({"ngram": phrase_text, "count": int(count)})

    return results


def _translation_metric(sample: Dict[str, Any], metric_name: str) -> Optional[float]:
    primary = sample.get("translation_difficulty", {}).get(metric_name)
    value = _safe_float(primary)
    if value is not None:
        return value

    legacy = sample.get("translation_difficult", {}).get(metric_name)
    return _safe_float(legacy)


def _collect_translation_metric(samples: List[Dict[str, Any]], metric_name: str) -> List[float]:
    out = []
    for sample in samples:
        value = _translation_metric(sample, metric_name)
        if value is not None:
            out.append(value)
    return out
