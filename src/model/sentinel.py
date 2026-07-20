from ext.guardians.sentinel_metric import download_model, load_from_checkpoint
from .scorer import Scorer
import torch

class SentinelScorer(Scorer):
    def __init__(self, device: str = "cuda"):
        super(SentinelScorer, self).__init__()

        if not torch.cuda.is_available() and 'cuda' in device:
            print(f"CUDA device {device} not found. Falling back to CPU.")
            self.device = 'cpu'
            self.gpus = 0
        else:
            print(f"Using device: {device} for SentinelScorer.")
            self.device = device
            self.gpus = 1


        # Directly load the checkpoint from the local path
        model_path = download_model("sapienzanlp/sentinel-ref-mqm")
        # model_path = "models/sentinel-ref-mqm/checkpoints/model.ckpt"
        self.model = load_from_checkpoint(model_path)

        self.model.to(self.device)

    def assign_score(self, input):
        data = [{"ref": el} for el in input]
        output = self.model.predict(data, batch_size=8, gpus=self.gpus)
        return output