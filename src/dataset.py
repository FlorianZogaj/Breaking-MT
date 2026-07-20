import json
from pathlib import Path
from typing import Dict, Any, Optional

from torch.utils.data import Dataset
from tqdm import tqdm

from prompts import MODIFICATION_PROMPT, NEW_PROMPT

class BaseDataset(Dataset):
    """
    Abstract base for all datasets.
    Each instance is of the form: id, original_sentence, prompt
    """

    def __init__(self, tokenizer, apply_chat_template: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template
        self.examples = []  

    def _build_prompt(self, sentence: str) -> Optional[str]:
        if not self.apply_chat_template:
            return sentence

        content = NEW_PROMPT.format(sentence=sentence)
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ex = self.examples[index]
        sentence = ex["original_sentence"]
        return {
            "id": ex["id"],
            "original_sentence": sentence,
            "prompt": self._build_prompt(sentence),
        }


class EnglishWmtSentencesDataset(BaseDataset):
    """Requirement: data/wmt19-23.jsonl or data/wmt{year}_sentences.jsonl"""

    def __init__(self, tokenizer, apply_chat_template: bool = True, year=None):
        super().__init__(tokenizer=tokenizer, apply_chat_template=apply_chat_template)

        data_path = Path("data/wmt19-23.jsonl") if year is None else Path(f"data/wmt{year}_sentences.jsonl")

        with open(data_path, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        desc = "Loading English WMT sentences" if year is None else f"Loading English WMT {year} sentences"
        for idx, el in enumerate(tqdm(raw, desc=desc)):
            text = str(el.get("text", "")).strip()
            if not text:
                continue

            item_year = str(el.get("year", year if year is not None else "unknown"))
            lang_pair = str(el.get("lang_pair", "unknown"))
            sentence_idx = el.get("index", idx)

            self.examples.append(
                {
                    "id": f"{item_year}_{lang_pair}_{sentence_idx}_{idx}",
                    "original_sentence": text,
                }
            )


class Wmt25Dataset(BaseDataset):
    """Requirement: data/wmt25-genmt-humeval.jsonl"""

    def __init__(
        self,
        tokenizer,
        apply_chat_template: bool = True,
        lan_source: str = "en",
        lan_target: str = "en",
    ):
        super().__init__(tokenizer=tokenizer, apply_chat_template=apply_chat_template)

        with open("data/wmt25-genmt-humeval.jsonl", "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]

        for idx, el in enumerate(tqdm(raw, desc="Loading WMT-25")):
            doc_id: str = el.get("doc_id", "")
            if doc_id[:2] == lan_source:
                self.examples.append(
                    {
                        "id": f"wmt25_src_{doc_id}_{idx}",
                        "original_sentence": el["src_text"],
                    }
                )
            if doc_id[3:5] == lan_target:
                self.examples.append(
                    {
                        "id": f"wmt25_tgt_{doc_id}_{idx}",
                        "original_sentence": el["tgt_text"],
                    }
                )
