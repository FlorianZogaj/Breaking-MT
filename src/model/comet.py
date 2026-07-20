from comet import download_model, load_from_checkpoint
from .scorer import Scorer
import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

class CometScorer(Scorer):
    def __init__(
        self,
        device: str = "cuda",
        nllb_model_id: str = "facebook/nllb-200-distilled-600M",
        helsinki_model_id: str = "Helsinki-NLP/opus-mt-en-it",
        translation_model: str = "average",
    ):
        super(CometScorer, self).__init__()
        valid_translation_models = {"average", "nllb", "helsinki"}
        if translation_model not in valid_translation_models:
            raise ValueError(
                f"translation_model must be one of {sorted(valid_translation_models)}, "
                f"got {translation_model!r}"
            )
        self.translation_model = translation_model

        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        self.model = load_from_checkpoint(model_path)
        
        self.nllb_model = pipeline(
            task="translation",
            model=nllb_model_id,
            src_lang="eng_Latn",
            tgt_lang="ita_Latn",
            torch_dtype=torch.float16,
            device=device,
            max_length=512,
        )

        self.helsinki_transl = pipeline(
            task="translation",
            model=helsinki_model_id,
            device=device,
            max_length=512,
            truncation=True,
        )

    def _predict_scores(self, data_score):
        model_output = self.model.predict(
            data_score,
            batch_size=8,
            gpus=1 if torch.cuda.is_available() else 0,
        )
        return np.array(model_output.scores)

    def assign_all_scores(self, src_text):
        result_nllb = self.nllb_model(src_text)
        nllb_mt = [d["translation_text"] for d in result_nllb]
        nllb_data = [{"src": s, "mt": m} for s, m in zip(src_text, nllb_mt)]

        result_helsinki = self.helsinki_transl(src_text)
        helsinki_mt = [d["translation_text"] for d in result_helsinki]
        helsinki_data = [{"src": s, "mt": m} for s, m in zip(src_text, helsinki_mt)]

        scores = self._predict_scores(nllb_data + helsinki_data)
        split = len(src_text)
        nllb_scores = scores[:split]
        helsinki_scores = scores[split:]

        return {
            "nllb": nllb_scores.tolist(),
            "helsinki": helsinki_scores.tolist(),
            "average": ((nllb_scores + helsinki_scores) / 2.0).tolist(),
        }

    def assign_score(self, src_text):
        if self.translation_model == "average":
            return self.assign_all_scores(src_text)["average"]

        if self.translation_model == "nllb":
            result_nllb = self.nllb_model(src_text)
            nllb_mt = [d["translation_text"] for d in result_nllb]
            nllb_data = [{"src": s, "mt": m} for s, m in zip(src_text, nllb_mt)]
            return self._predict_scores(nllb_data).tolist()

        result_helsinki = self.helsinki_transl(src_text)
        helsinki_mt = [d["translation_text"] for d in result_helsinki]
        helsinki_data = [{"src": s, "mt": m} for s, m in zip(src_text, helsinki_mt)]
        return self._predict_scores(helsinki_data).tolist()
        
