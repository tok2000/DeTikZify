import torch
import torch.nn as nn
from open_clip import create_model, get_tokenizer, image_transform
from timm.models.convnext import ConvNeXt
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import os

from .base_encoder import BaseVisionTower, ProcessorWrapper


def extract_res_interp(model_name):
    valid_model_prefixes = {
        "CLIP-convnext-L":"/home/tikrause/.cache/huggingface/hub/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/snapshots/654d0f80ff73c58e7281a3ca7dc425589049e2e1",
        "CLIP-convnext-XXL":"/home/tikrause/.cache/huggingface/hub/models--laion--CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/snapshots/9f3e8ee3f383c672388d9178afe70af9e63ac9df"
    }


    res = None
    interp = None

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class CLIPConvNextTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        """
        Initialize the CLIPConvNextTower.

        Args:
            vision_tower (str): The name of the vision tower model in the format "clip-convnext-resXXX-interpYYY".
            args (argparse.Namespace): The arguments parsed from the command line.
            delay_load (bool, optional): Whether to delay loading the model. Defaults to False.
        """
        super().__init__(vision_tower, args, delay_load)

        self.is_multi_stage = "multi-stage" in vision_tower
        base_model_name, res, interp = extract_res_interp(vision_tower)
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 1024
        #self._interp_size = interp if interp is not None else 729  # default 729
        self._reduction = 32

        self.select_layer = getattr(args, "mm_vision_select_layer", 12)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.is_loaded = False

        if not delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            assert "CLIP-convnext-L" in vision_tower or "CLIP-convnext-XXL" in vision_tower
            if "CLIP-convnext-L" in vision_tower:
                if "multi-stage" in vision_tower:
                    self._hidden_size = sum([192, 384, 768, 1536])
                else:
                    self._hidden_size = 1536
            else:
                if "multi-stage" in vision_tower:
                    self._hidden_size = sum([384, 768, 1536, 3072])
                else:
                    self._hidden_size = 3072

    def load_model(self, device_map=None):
        """
        Load the CLIP-ConvNext model.
        """
        assert "convnext" in self.vision_tower_name.lower()
        self.vision_model = "convnext"
    
        local_model_path = self.vision_tower_name

        if "xxl" in local_model_path.lower() or "CLIP-convnext-XXL" in self.vision_tower_name:
            model_name = "convnext_xxlarge_320"
            hidden_size = 3072
        elif "large" in local_model_path.lower() or "CLIP-convnext-L" in self.vision_tower_name:
            model_name = "convnext_large_d_320"
            hidden_size = 1536
        else:
            raise ValueError(f"Unknown convnext variant in path: {local_model_path}")

        model_bin = os.path.join(local_model_path, "open_clip_pytorch_model.bin")
        if os.path.exists(model_bin):
            self.vision_tower = create_model(
                model_name,  # Adjust if needed
                pretrained=model_bin,
                precision="fp32",
            ).visual
    
            transform = image_transform(
                image_size=self._image_size,
                is_train=False
            )
            self.image_processor = ProcessorWrapper(transform, height=self._image_size, width=self._image_size)
            self.tokenizer = get_tokenizer(model_name)
            self._hidden_size = hidden_size
    
        else:
            clip_model = AutoModel.from_pretrained(self.vision_tower_name)
            processor = AutoProcessor.from_pretrained(self.vision_tower_name)
    
            self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)
            self.vision_tower = clip_model.vision_model  # Assign the vision model part
    
            if hasattr(self.vision_tower, "feature_info"):
                feature_info = self.vision_tower.feature_info
                self._hidden_size = sum([stage['num_chs'] for stage in feature_info]) if self.is_multi_stage else feature_info[-1]['num_chs']
            else:
                self._hidden_size = hidden_size
    
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def interpolate(self, image_forward_outs):
        """
        Interpolate the image features to the desired number of patches.

        Args:
            image_forward_outs (torch.Tensor): The output features from the vision tower.

        Returns:
            torch.Tensor: The interpolated image features.
        """
        if self._interp_size is None:
            return image_forward_outs

        image_features = F.interpolate(
            image_forward_outs.float(),
            size=(self.num_patches_per_side, self.num_patches_per_side),
            mode='bilinear',
            align_corners=False
        ).to(dtype=image_forward_outs.dtype)
        image_features = image_features.flatten(2, 3).permute(0, 2, 1).contiguous()
        return image_features

    def _forward(self, images, interp = True):
        """
        Perform the forward pass of the CLIPConvNextTower.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The output features from the vision tower after interpolation.
        """
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            #print("[DEBUG]: ConvNext Tower is used.")
            image_features = self.vision_tower.trunk.forward_features(images.to(device=self.device, dtype=self.dtype))  # [B, C, H, W]

            if interp:
                image_features = self.interpolate(image_features)
            
            if image_features.ndim == 2:
                # [B, C] → [B, 1, C]
                image_features = image_features.unsqueeze(1)
            elif image_features.ndim == 4:
                # [B, C, H, W] → [B, H*W, C]
                image_features = image_features.flatten(2).transpose(1, 2)

            return image_features

    @property
    def image_size(self):
        return self._image_size

    @property
    def num_patches_per_side(self):
        """
        Get the number of patches per side.

        Returns:
            int: The number of patches per side.
        """
        if self._interp_size is None:
            return self._image_size // self._reduction
        else:
            return int(self._interp_size ** 0.5)

    @property
    def num_patches(self):
        """
        Get the total number of patches.

        Default: 256

        Returns:
            int: The total number of patches.
        """
        if self._interp_size is None:
            return (self._image_size // self._reduction) ** 2
        else:
            return self._interp_size

    def get_output_grid_shape(self, image_size: int = 384):
        dummy = torch.randn(1, 3, image_size, image_size).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output = self._forward(dummy)
            seq_len = output.shape[1]
            grid_size = int(seq_len**0.5)
            return (grid_size, grid_size)
