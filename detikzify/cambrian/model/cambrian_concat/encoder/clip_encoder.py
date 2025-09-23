import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip import create_model, get_tokenizer, image_transform, create_model_and_transforms
from ezcolorlog import root_logger as logger

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from .base_encoder import BaseVisionTower, ProcessorWrapper
from ..utils import IS_XLA_AVAILABLE


def extract_interp(model_name):
    interp = None
    res = None
    base_model_name = model_name

    if "interp" in model_name:
        base_model_name = model_name.split('-interp')[0]

    # Parse resolution from model name
    parts = model_name.split("-")
    for part in parts:
        if part.startswith("interp"):
            interp = int(part[6:])
        elif part.isdigit() and len(part) == 3:  # e.g., "336", "224"
            res = int(part)
    
    # Default resolution based on model type
    if res is None:
        if "large" in model_name.lower():
            res = 336
        else:
            res = 224

    return base_model_name, res, interp


class ClipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, res, interp = extract_interp(vision_tower_name) # initialize the base model name and interpolation size
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 224
        
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            logger.debug(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return
        
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower) # enable gradient computation for the vision model
        self.is_loaded = True

        if IS_XLA_AVAILABLE:
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

    def _feature_select(self, image_features):
        if self.select_feature == 'patch':
            features = image_features[:, 1:] # Skip the CLS token
        elif self.select_feature == 'cls_patch':
            features = image_features # Include the CLS token
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return self._feature_select(image_features)

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape # get batch size, number of tokens, and feature dimension

        # Only interpolate if the number of tokens does not match the expected number of patches
        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images, interp = True):
        if IS_XLA_AVAILABLE:
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            # Resize to the model's expected input size
            if images.shape[-1] != self._image_size:
                images = F.interpolate(images, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False)

            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            if isinstance(image_features, tuple):
                image_features = image_features[1] # Take the last layer if multiple are returned
            
            if interp:
                image_features = self.interpolate(image_features)
            return image_features

    # compute the number of patches based on image size and patch size
    def get_output_grid_shape(self, image_size: int = 384):
        dummy = torch.randn(1, 3, image_size, image_size).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output = self._forward(dummy)
            seq_len = output.shape[1]
            grid_size = int(seq_len**0.5)
            return (grid_size, grid_size)
