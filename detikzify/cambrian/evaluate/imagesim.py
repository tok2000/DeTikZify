from functools import cached_property
from math import tanh
from typing import List, Literal

from PIL import Image
from ot.lp import emd2
import torch
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import AutoModel, PreTrainedModel, AutoImageProcessor, ProcessorMixin

from ..util import expand, infer_device, load

class ImageSim(Metric):
    """Perceptual image similarity using visual encoders."""

    higher_is_better = True

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        mode: Literal["emd", "cos", "cos_avg"] = "emd",
        preprocess: bool = True,
        device: str = infer_device(),
        dtype=torch.bfloat16 if is_cuda_available() and is_bf16_supported() else torch.float16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.preprocess = preprocess
        self.mode = mode
        self._device = device
        self.set_dtype(dtype)

        self.add_state("score", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def __str__(self):
        return self.__class__.__name__ + f" ({self.mode.upper().replace('_', '-')})"

    @cached_property
    def models(self) -> List[torch.nn.Module]: # changed method to enable multi-tower architecture
        model_list = []
        for name in self.model_name if isinstance(self.model_name, list) else [self.model_name]:
            name = name.replace("hf-hub:", "")
            model = AutoModel.from_pretrained(name, torch_dtype=self.dtype)
            model = model.vision_model.to(self.device)
            model_list.append(model)
        return model_list

    @cached_property
    def processors(self) -> List[ProcessorMixin]: # changed method to enable multi-tower architecture
        return [AutoImageProcessor.from_pretrained(name) for name in (self.model_name if isinstance(self.model_name, list) else [self.model_name])]

    @classmethod
    def from_detikzify(cls, model: PreTrainedModel, processor: ProcessorMixin, *args, **kwargs): # changed method to enable multi-tower architecture
        vision_towers = model.model.vision_models
        processors = [tower.image_processor for tower in vision_towers]
    
        imagesim = cls(*args, **(kwargs | {
            "model_name": [getattr(tower, "vision_tower_name", "unknown") for tower in vision_towers]
        }))
        imagesim.models = [tower.vision_tower for tower in vision_towers]
        imagesim.processors = processors
        return imagesim

    def get_vision_features(self, image: Image.Image | str): # changed method to enable multi-tower architecture
        image = load(image)
        if self.preprocess:
            image = expand(image, max(image.size), do_trim=True)
    
        features = []
        with torch.inference_mode():
            for model, processor in zip(self.models, self.processors):
                encoding = processor(image, return_tensors="pt").to(self.device, self.dtype)
                if self.mode == "cos":
                    feat = model(**encoding).pooler_output.squeeze()
                elif self.mode == "cos_avg":
                    feat = model(**encoding).last_hidden_state.squeeze().mean(dim=0)
                else:
                    feat = model(**encoding).last_hidden_state.squeeze()
                features.append(feat)
        
        # example aggregation: mean
        return torch.stack(features).mean(dim=0)

    def get_similarity(self, img1: Image.Image | str, img2: Image.Image | str):
        img1_feats = self.get_vision_features(img1)
        img2_feats = self.get_vision_features(img2)

        if img1_feats.is_mps: # mps backend does not support dtype double
            img1_feats, img2_feats = img1_feats.cpu(), img2_feats.cpu()
        if img1_feats.ndim > 1:
            dists = 1 - pairwise_cosine_similarity(img1_feats.double(), img2_feats.double()).cpu().numpy()
            return 2 * tanh(-emd2(M=dists, a=list(), b=list())) + 1 # type: ignore
        else:
            return F.cosine_similarity(img1_feats.double(), img2_feats.double(), dim=0).item()

    def update(
        self,
        img1: Image.Image | str | List[Image.Image | str],
        img2: Image.Image | str | List[Image.Image | str],
    ):
        if isinstance(img1, List) or isinstance(img2, List):
            assert type(img1) == type(img2) and len(img1) == len(img2) # type: ignore
        else:
            img1, img2 = [img1], [img2]

        for pair in zip(img1, img2): # type: ignore
            self.score += self.get_similarity(*pair)
            self.n_samples += 1

    def compute(self):
        return (self.score / self.n_samples).item()
