import random
import re 
import math
import language_tool_python
import wandb

from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils import (
    RewardWeights, 
    word_count,
    Embedder,
    AcceptabilityClassifier,
    RolloutWandbLogger,
    RawDeltaAccumulator,
)

from model.comet import CometScorer
from model.sentinel import SentinelScorer


class RewardComputer:
    def __init__(self, 
                 weights: RewardWeights,
                 td_estimator: str,
                 comet_translation_model: str,
                 embedder_name: str,
                 cola_name: str, 
                 rollout_logger: Optional[RolloutWandbLogger],
                 raw_delta_accumulator: Optional[RawDeltaAccumulator],
                 device: str):
        self.weights = weights
        self.td_estimator = td_estimator
        self.rollout_logger = rollout_logger
        self.raw_delta_accumulator = raw_delta_accumulator

        if self.td_estimator == "sentinel": 
            self.sentinel = SentinelScorer()
        elif self.td_estimator == "comet":
            self.comet = CometScorer(device=device, translation_model=comet_translation_model)
        else:
            raise ValueError(f"{self.td_estimator} not implemented yet")
        
        # for semantic similarity
        self.embedder = Embedder(model_name=embedder_name, device=device)

        # for grammatical correctness
        self.ltp_tool = language_tool_python.LanguageTool('en-US')
        self.gc_classififier = AcceptabilityClassifier(model_name=cola_name, device=device)
        
        self.step = 0
        
    def __call__(self,
                 completions, 
                 original_sentence,
                 **kwargs) -> List[float]:
        
        generations = self._extract_generations(completions)

        # === Get rewards

        # - Format
        r_format_strict, r_format_soft, r_format_tag_pen = self.reward_format(generations)

        # - Translation difficulty
        r_diff_abs, r_diff_delta, r_raw_delta = self.reward_translation_difficulty(generations, original_sentence)

        # - Generation control
        r_len, r_single_sen, r_rep_pen, r_illegal_chars = self.reward_generation_control(generations, original_sentence)

        # - Semantic similarity
        r_semantic_sim = self.reward_semantic_similarity(generations, original_sentence) 

        # - Grammatical correctness
        r_grammatical_ltp, r_grammatical_cola = self.reward_grammatical_correctness(generations)

        # - Total
        reward_dict = {
            "td_diff_abs": r_diff_abs,
            "td_diff_delta": r_diff_delta,
            "td_raw_delta": r_raw_delta,
            "format_strict": r_format_strict,
            "format_soft": r_format_soft,
            "format_tag_pen": r_format_tag_pen,
            "len": r_len,
            "single_pen": r_single_sen,
            "rep_pen": r_rep_pen,
            "illegal_chars": r_illegal_chars,
            "semantic_sim": r_semantic_sim,
            "grammatical_ltp": r_grammatical_ltp,
            "grammatical_cola": r_grammatical_cola,
        }
        r_totals = []
        for i in range(len(generations)):
            total_i = 0.0
            for name, values in reward_dict.items():
                if name == "td_raw_delta":
                    continue
                weight = getattr(self.weights, name)
                total_i += weight * float(values[i])
            r_totals.append(total_i)   
        reward_dict["total"] = r_totals  

        trainer = kwargs.get("trainer", None)
        self.log_rollouts(
            rollout_logger=self.rollout_logger,
            step=self.step,
            trainer=trainer,
            completions=completions,
            generations=generations,
            originals=original_sentence,
            rewards=reward_dict,
            log_random_k=kwargs.get("log_random_k", 8),
        )

        self.step += 1
        return r_totals

    def _extract_generations(self, completions) -> List[str]:
        """Extracts the generated complex sentence from the model output"""
        # def extract_rewrite(text: str) -> str:
        #     # 1)Ideal case: <rewrite> ... </rewrite>
        #     REWRITE_SOFT_RE = re.compile(r"<rewrite>\s*(.*?)\s*</rewrite>", re.DOTALL)
        #     m = REWRITE_SOFT_RE.search(text)
        #     if m:
        #         s = re.sub(r"\s+", " ", m.group(1).strip())
        #         return s

        #     t = (text or "").strip()

        #     # 2) Common failure: starts with </rewrite> then the sentence
        #     t = re.sub(r"^\s*</rewrite>\s*", "", t)

        #     # 3) Has opening tag but missing/garbled closing tag
        #     if "<rewrite>" in t:
        #         after = t.split("<rewrite>", 1)[1].strip()
        #         after = after.split("</rewrite>", 1)[0].strip()
        #         after = re.sub(r"\s+", " ", after)
        #         return after

        #     # 4) No tags at all: fall back to first non-empty line
        #     first_line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
        #     first_line = re.sub(r"\s+", " ", first_line)
        #     return first_line

        # print(completions)
        # print(completions[0])
        # print(completions[0]["content"])
        print(f"[DEBUG] type(completions): {type(completions)}")
        print(f"[DEBUG] type(completions[0]): {type(completions[0])}")
        print(f"[DEBUG] completions[0]: {repr(completions[0])[:300]}")

        responses = [c for c in completions]
        # generations = [extract_rewrite(r) for r in responses]
        generations = responses
        return generations
    
    # == Formatting issues 
    def reward_format(self, generations):
        """Computes format reward, as a combination of format_strict, format_soft, and format_tag_order_pen"""
        rew_strict = -0.2
        rew_soft = -0.2
        rew_tag_order_pen = -0.2

        n = len(generations)
        r_format_strict = [0.0] * n
        r_format_soft = [0.0] * n
        r_format_tag_pen = [0.0] * n
        
        for i, generation in enumerate(generations):
            if not generation:
                continue

            # --- strict format
            REWRITE_STRICT_RE = re.compile(r"^<rewrite>\n([^\n]+)\n</rewrite>\n?$")
            if REWRITE_STRICT_RE.match(generation) is not None:
                r_format_strict[i] = rew_strict

            # --- soft format
            REWRITE_SOFT_RE = re.compile(r"<rewrite>\s*(.*?)\s*</rewrite>", re.DOTALL)
            if REWRITE_SOFT_RE.search(generation):
                r_format_soft[i] = rew_soft

            # --- tag order penalty
            low = generation.lower()
            a = low.find("<rewrite>")
            b = low.find("</rewrite>")
            if b != -1 and (a == -1 or b < a):
                r_format_tag_pen[i] = rew_tag_order_pen

        return r_format_strict, r_format_soft, r_format_tag_pen
    
    # == Translation difficulty 
    def reward_translation_difficulty(self, generations, originals):
        """Computes translation difficulty reward, either using sentinel or comet"""
        if self.td_estimator == "sentinel":
            reward_fnc = self._reward_sentinel
            diff_clip = 2.0
        elif self.td_estimator == "comet":
            reward_fnc = self._reward_comet
            diff_clip = 2.0
        else:
            raise ValueError(f"The translation estimator {self.td_estimator} not yet implemented")
        
        # get scores singular
        r_generations = reward_fnc(generations)
        r_originals = reward_fnc(originals)

        # get scores difference
        r_diff = []
        r_raw_delta = []
        for o, g in zip(r_originals, r_generations):
            raw_delta = o - g    
            scaled_delta = raw_delta * 5.0
            d = max(-diff_clip, min(diff_clip, scaled_delta))
            r_diff.append(d)
            r_raw_delta.append(raw_delta)
        
        return r_generations, r_diff, r_raw_delta


    def _reward_sentinel(self, texts):
        """Evaluates generated sentences using sentinel"""
        # keep valid inputs
        scores = [0.0] * len(texts)
        idxs = [i for i, t in enumerate(texts) if t]
        if not idxs:
            return scores
        clean_texts = [texts[i] for i in idxs] 

        sentinel_scores = self.sentinel.assign_score(clean_texts)
        for i, sc in zip(idxs, sentinel_scores["scores"]):
            scores[i] = float(sc)

        return scores
                
    def _reward_comet(self, texts):
        """Evaluates generated sentences using comet"""
        # keep valid inputs
        scores = [0.0] * len(texts)
        idxs = [i for i, t in enumerate(texts) if t]
        if not idxs:
            return scores
        clean_texts = [texts[i] for i in idxs] 

        comet_scores = self.comet.assign_score(clean_texts)
        for i, sc in zip(idxs, comet_scores):
            scores[i] = float(sc)

        return scores
    
    # == Generation control
    def reward_generation_control(self, generations, originals):
        # keep valid inputs
        base_r_len = [0.0] * len(generations)
        base_r_single_pen = [0.0] * len(generations)
        base_r_rep_pen = [0.0] * len(generations)
        base_r_illegal_chars = [0.0] * len(generations)
        idxs = [i for i, t in enumerate(generations) if t]
        if not idxs:
            return base_r_len, base_r_single_pen, base_r_single_pen, base_r_illegal_chars
        clean_texts = [generations[i] for i in idxs] 
        clean_originals = [originals[i] for i in idxs]

        r_len = self._reward_lenght(clean_texts, clean_originals)
        r_single_sen = self._reward_single_sentence(clean_texts)
        r_rep_pen = self._reward_repetition_penalty(clean_texts)
        r_illegal_chars = self._reward_illegal_characters(clean_texts)
        
        for i, rl, rss, rrp, ric in zip(idxs, r_len, r_single_sen, r_rep_pen, r_illegal_chars):
            base_r_len[i] = float(rl)
            base_r_single_pen[i] = float(rss)
            base_r_rep_pen[i] = float(rrp)
            base_r_illegal_chars[i] = float(ric)

        return base_r_len, base_r_single_pen, base_r_rep_pen, base_r_illegal_chars
    
    def _reward_lenght(self, texts, originals):
        """Computes reward for relative length"""
        r_len = []
        for text, original in zip(texts, originals):
            ow = max(1, word_count(original))
            rwc = max(1, word_count(text))
            ratio = rwc / ow

            if ratio < 0.5 or ratio > 2.0:
                r_len.append(-2)
            else:
                dist = abs(math.log(ratio))
                shaped = max(0.0, 1.0 - dist / math.log(2.0))
                r_len.append(shaped)

        return r_len

    def _reward_single_sentence(self, texts):
        """Computes reward for producing just a single sentence"""
        r_single_pen = []
        for text in texts:
            # count num of sentences
            ss = re.sub(r"\b(e\.g|i\.e|mr|mrs|dr)\.", r"\1", text, flags=re.IGNORECASE)
            n = len(re.findall(r"[.!?]", ss))
            r_single_pen.append(-0.3 if n <= 1 else 0.0)

        return r_single_pen


    def _reward_repetition_penalty(self, texts):
        """Computes reward for too much repetition"""
        def repetition_penalty_scalar(s: str) -> float:
            if not s:
                return 0.0
            if re.search(r"(.)\1\1\1\1", s):
                return 1.0
            toks = re.findall(r"\b\w+\b", s.lower())
            if len(toks) >= 12:
                unique = len(set(toks))
                if unique / len(toks) < 0.5:
                    return -0.5
            return 0.0
        
        r_rep_pen = []
        for text in texts:
            rep = repetition_penalty_scalar(text)
            r_rep_pen.append(rep)

        return r_rep_pen

    def _reward_illegal_characters(self, texts):
        """Computes reward for illegal characters"""
        r_illegal_chars = []
        for text in texts:
            penalty = 0
            if not text.isascii():
                penalty = -0.5                
            r_illegal_chars.append(penalty)

        return r_illegal_chars

    

    # == Semantic similarity
    def reward_semantic_similarity(self, generations, originals):
        """Computes reward for semantic similarity"""    
        r_sim = [0.0] * len(generations)
        idxs = [i for i, t in enumerate(generations) if t]
        if not idxs:
            return r_sim
        

        s_floor = 0.70
        ss = self.embedder.cosine_sim([originals[i] for i in idxs], [generations[i] for i in idxs])
        for i, s in zip(idxs, ss):
            s = float(s)
            if s < s_floor:
                r_sim[i] = -0.5
            else:
                #shaped = (s - s_floor) / max(1e-6, (1.0 - s_floor))
                #r_sim[i] = shaped
                r_sim[i] = 0.5 # not shaped, just a positive reward for being above the threshold 
        return r_sim

    
    # == Grammatical correctness
    def reward_grammatical_correctness(self, generations):
        r_grammatical_ltp = self._reward_grammatical_ltp(generations)
        r_grammatical_cola = self._reward_grammatical_cola(generations)
        return r_grammatical_ltp, r_grammatical_cola

    
    def _reward_grammatical_cola(self, texts):
        """Computes reward grammatical correctness via cola"""
        # keep valid scores
        scores = [0.0] * len(texts)
        idxs = [i for i, t in enumerate(texts) if t]
        if not idxs:
            return scores
        
        ps = self.gc_classififier.p_acceptable([texts[i] for i in idxs])
        for i, p in zip(idxs, ps):
            scores[i] = (float(p) - 0.5) * 2.0

        return scores
    
    def _reward_grammatical_ltp(self, texts):
        """Computes reward grammatical correctness via language tools"""
        # keep valid scores
        scores = [0.0] * len(texts)
        idxs = [i for i, t in enumerate(texts) if t]
        if not idxs:
            return scores
        
        for i in idxs:
            text = texts[i]
            try:
                matches = self.ltp_tool.check(text)
                error_count = len(matches)

                reward = 1.0 - 2* math.log1p(error_count)
                scores[i] = max(-2, reward)
            except Exception:
                scores[i] = 0.0
        return scores

    def log_rollouts(
        self,
        rollout_logger: Any,
        *,
        step: int,
        trainer: Optional[Any],
        completions: List[str],
        generations: List[str],
        originals: List[str],
        rewards: Dict[str, List[float]],
        log_random_k: int = 4,
        **kwargs
    ) -> None:
        """Logs"""

        is_main = True
        log_step = int(step)
        if trainer is not None and hasattr(trainer, "accelerator"):
            is_main = bool(trainer.accelerator.is_main_process)
        if trainer is not None and hasattr(trainer, "state") and getattr(trainer.state, "global_step", None) is not None:
            log_step = int(trainer.state.global_step)
        if not is_main:
            return

        # is logger available
        if rollout_logger is None or not getattr(rollout_logger, "enabled", False):
            return

        raw_deltas = rewards.get("td_raw_delta")
        if raw_deltas and self.raw_delta_accumulator is not None:
            self.raw_delta_accumulator.add(raw_deltas)

        # pick some candidates 
        has_gen = [bool(g) for g in generations]
        candidates = list(range(len(generations)))
        candidates.sort(key=lambda i: 0 if has_gen[i] else 1)
        if sum(has_gen) > log_random_k:
            good = [i for i in candidates if has_gen[i]]
            chosen = random.sample(good, log_random_k)
        else:
            chosen = candidates[:log_random_k]
        
        rows = []
        for i in chosen:
            row = {
                "step": log_step,
                "original": originals[i] if i < len(originals) else "",
                "generation": generations[i],
                "raw_completion": (completions[i] if i < len(completions) else "")[:2000],
            }
            for name, values in rewards.items():
                if isinstance(values, list):
                    row[name] = float(values[i])
            rows.append(row)

        rollout_logger.add_many(rows)
