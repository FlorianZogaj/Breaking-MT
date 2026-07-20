# Breaking-MT: Generating Adversarial Texts for Machine Translation via GRPO

Code for the paper **"Generating Adversarial Texts for Machine Translation via GRPO."**

As modern machine translation (MT) systems improve, standard benchmarks become
less useful for exposing their remaining weaknesses. This project fine-tunes a
large language model with **Group Relative Policy Optimization (GRPO)** to rewrite
existing English source sentences into variants that are **harder to translate**,
while preserving meaning, grammaticality, and approximate length. On WMT25 the
approach lowers average COMET translation quality from **0.63 → 0.48** (the
non-fine-tuned base model stays at 0.64), and the effect transfers to the unseen
WMT19–WMT24 benchmarks.

The pipeline has two stages:

- **Training** — the LLM proposes several rewrites per source sentence; each is
  scored by a translation-difficulty estimator plus linguistic-constraint rewards,
  and GRPO increases the likelihood of the higher-reward rewrites.
- **Inference / evaluation** — the fine-tuned model rewrites source text directly,
  and the rewrites are scored on translation difficulty, grammaticality,
  complexity, diversity, and an LLM-as-judge.

---

## Repository structure

```
configs/
  training_config.yaml     # GRPO fine-tuning configuration
  eval_config.yaml         # Evaluation configuration
src/
  model_training.py        # Training entry point
  evaluate.py              # Evaluation entry point
  dataset.py               # WMT25 dataset loading + rewrite prompting
  prompts.py               # Rewrite prompt + LLM-as-judge prompt
  rewards.py               # GRPO reward computation
  training_config.py       # Training arg parsing / config merge
  evaluate_config.py       # Evaluation arg parsing / config merge
  evaluate_scores.py       # Evaluation metrics (difficulty, complexity, judge, ...)
  evaluate_utils.py        # Evaluation helpers
  utils.py                 # Shared utilities (dataset loading, embedder, CoLA, ...)
  model/                   # Difficulty estimators: COMET (MT+QE) and Sentinel
  ext/guardians/           # Vendored Sentinel metric (third party — see License)
download_data.sh           # Downloads WMT25
environment.yml            # Conda environment
```

---

## Setup

The project uses **Conda** for dependencies and **[Unsloth](https://github.com/unslothai/unsloth)**
for efficient LoRA fine-tuning. A CUDA-capable GPU is required.

### 1. Create the environment

`environment.yml` pins both the Conda and Pip dependencies, including a Java
runtime (`openjdk`) that `language_tool_python` needs for the grammaticality
reward — so no system-level Java or module loading is required.

```bash
conda env create -f environment.yml
conda activate mtbreaker
```

> **Always activate the environment before running** — the grammar checker looks
> for `java` on your `PATH`, which activation provides.

### 2. Authenticate with Hugging Face

Several models are gated and require accepting their terms on the Hugging Face Hub
and logging in:

```bash
hf auth login
```

Gated models used here:
- `meta-llama/Meta-Llama-3.1-8B-Instruct` — the rewriting model **and** the LLM judge
- `Unbabel/wmt22-cometkiwi-da` — the COMET-Kiwi quality-estimation model

### 3. Download the data

```bash
./download_data.sh
```

This fetches the WMT25 general-MT human-evaluation set to
`data/wmt25-genmt-humeval.jsonl`.

> **Run all scripts from the repository root.** Data paths (e.g. `data/...`) are
> resolved relative to the current working directory, so commands assume you are
> in the repo root with the environment activated.

---

## Training (GRPO fine-tuning)

```bash
python src/model_training.py --config configs/training_config.yaml
```

This fine-tunes `Llama-3.1-8B-Instruct` with LoRA using GRPO. Run
`python src/model_training.py --help` to see all overridable parameters; anything
in the config can also be passed on the command line (CLI flags take precedence).

The trained LoRA adapter is written to `output_dir` (default
`./grpo_harder_translate_lora`, which is git-ignored). Weights & Biases logging is
enabled by default (`wandb: true`); disable it with `wandb: false` in the config,
or place your key in a `wandb.key` file at the repo root to log in automatically.

### Translation-difficulty reward

The main training signal is the **change in estimated translation difficulty**
between the original sentence and its rewrite. Two estimator families are
supported via `td_estimator`:

| `td_estimator` | Estimator | Notes |
| --- | --- | --- |
| `comet` | COMET-Kiwi (MT + QE) | Translates the rewrite, then scores it reference-free. Pick the MT system with comet_translation_model: nllb, helsinki, average. |
| `sentinel` | Sentinel | Estimates difficulty from the source sentence alone (no translation needed). |

Alongside the difficulty reward, constraint rewards preserve quality: approximate
**length** preservation, **semantic similarity** (MiniLM cosine), and
**grammaticality** (a RoBERTa CoLA classifier and LanguageTool error counts). The
reward formulation and weights follow Section 3.3 of the paper and are exposed as
`w_*` keys in the config.

> **Checkpoint selection matters.** As training continues, the model tends to
> collapse onto a few high-reward constructions. The paper selects an intermediate
> checkpoint (step 2000) that best balances difficulty, readability, and diversity
> rather than the one with the lowest translation quality. Monitor grammaticality,
> naturalness, and repeated-bigram frequency when choosing a checkpoint.

---

## Evaluation

```bash
python src/evaluate.py --config configs/eval_config.yaml
```

Key options in
`eval_config.yaml`:

- **`model`** — path to the fine-tuned generation model (LoRA adapter directory) to
  evaluate. Leave it `null` to score the **original source sentences** as a
  baseline (no generation model is loaded).
- **`model_judge`** — the LLM-as-judge (`Llama-3.1-8B-Instruct`). Set it to `null`
  to skip the judge-based metrics.
- **`n_samples`** — number of samples to evaluate (`-1` for all).

The pipeline scores each rewrite on:

- **Translation difficulty** — Sentinel and COMET (NLLB + Helsinki) scores
- **Grammaticality** — LanguageTool error counts
- **Complexity/readability** — RIX, entropy, word/sentence length (textdescriptives)
- **Diversity** — pairwise self-chrF and top lemmatized n-grams
- **LLM-as-judge** — naturalness, word rarity, syntactic complexity, and topics

Outputs are written under `path_to_save` (default `./evaluation/`):
`samples.jsonl` (per-sample scores), `metric_report.txt` (readable per-sample
report), and `summary.json` (aggregated metrics).

---

## Third-party code and attribution

`src/ext/guardians/` vendors the **Sentinel** translation-difficulty metric
(`sentinel_metric`). It is **not** covered by this repository's license: it retains
its original license, **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
(CC BY-NC-SA 4.0)**, as stated in [`src/ext/guardians/LICENSE.txt`](src/ext/guardians/LICENSE.txt).
Note the **non-commercial** restriction on this component.

This project also builds on: COMET-Kiwi (Rei et al., 2022), NLLB
(Costa-Jussà et al., 2022), Opus-MT / Helsinki (Tiedemann and Thottingal, 2020),
a RoBERTa CoLA classifier (Warstadt et al., 2019), LanguageTool, textdescriptives
(Hansen et al., 2023), and Unsloth.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{breaking-mt,
  title     = {Generating Adversarial Texts for Machine Translation via GRPO},
author    = {Zogaj, Florian and Hüttenender, Jakob and De Muri, Giovanni and
               Villa, Federico and Sood, Aryan and Zouhar, Vilém},
  year      = {2026}
}
```

> The venue/booktitle will be finalized in the camera-ready version.

---

## License

This repository's own code is released under the [MIT License](LICENSE), **with the
exception of `src/ext/guardians/`**, which is licensed separately under
CC BY-NC-SA 4.0 (see [Third-party code and attribution](#third-party-code-and-attribution)).
