import torch
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_model_config, CLIP
from transformers import AutoProcessor
from huggingface_hub import hf_hub_download
import safetensors.torch as safe_torch

from .base_encoder import ProcessorWrapper
from .clip_encoder import ClipVisionTower

def extract_res_interp(model_name):
    valid_model_prefixes = {
        "siglip/CLIP-ViT-SO400M-14-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "timm/ViT-SO400M-14-SigLIP-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "siglip/CLIP-ViT-SO400M-14":"hf-hub:timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP":"hf-hub:timm/ViT-SO400M-14-SigLIP",
        "google/siglip-so400m-patch14-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "hf-hub:timm/ViT-SO400M-14-SigLIP-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
    }

    res = 384 if '384' in model_name else 224
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

class SiglipVisionTower(ClipVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, res, interp = extract_res_interp(vision_tower_name)
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        
        model_name = self.vision_tower_name

        try:
            # Try the original Cambrian-1 approach first
            clip_model, processor = create_model_from_pretrained(self.vision_tower_name)
            
            self.vision_tower = clip_model.visual.trunk
            self.vision_tower.output_tokens = True
            
            self._hidden_size = self.vision_tower.embed_dim
            self._image_size = self.vision_tower.patch_embed.img_size[0]
            self._patch_size = self.vision_tower.patch_embed.patch_size[0]
            self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)
            
        except Exception as e:
            # Fallback to more complex approach if needed
            print(f"Warning: Simple SigLIP loading failed: {e}")
            print("Falling back to manual loading approach...")
            
            hf_to_open_clip = {
                "google/siglip-so400m-patch14-384": "ViT-SO400M-14-SigLIP-384",
                "siglip/CLIP-ViT-SO400M-14-384": "ViT-SO400M-14-SigLIP-384",
                "siglip/CLIP-ViT-SO400M-14": "ViT-SO400M-14-SigLIP",
                "timm/ViT-SO400M-14-SigLIP": "ViT-SO400M-14-SigLIP",
            }
            open_clip_name = hf_to_open_clip.get(model_name, model_name).replace("hf-hub:timm/", "") # map to open_clip name

            checkpoint_path = hf_hub_download(repo_id="timm/ViT-SO400M-14-SigLIP-384", filename="open_clip_model.safetensors")

            model_config = get_model_config(open_clip_name)
            model_config.pop("custom_text", None)

            model = CLIP(**model_config)
            
            full_state_dict = safe_torch.load_file(checkpoint_path)

            vision_state_dict = {k.replace("visual.", ""): v for k, v in full_state_dict.items() if k.startswith("visual.")}

            self.vision_tower = model.visual.trunk
            self.vision_tower.output_tokens = True
            self._hidden_size = self.vision_tower.embed_dim
            self._image_size = self.vision_tower.patch_embed.img_size[0]
            self._patch_size = self.vision_tower.patch_embed.patch_size[0]
            self.image_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True
        
    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

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

    def _forward(self, images, interpolate_token = 576, interp = True):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            assert images.dim() == 4, f"Expected image input of shape [B, C, H, W], but got {images.shape}" # check input shape
            image_features = self.vision_tower.forward_features(images.to(self.device, dtype=self.dtype))
            
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(1) # [B, C] → [B, 1, C]
            elif image_features.size(1) == 1:
                raise ValueError("SigLIP is only returning a single token — patch features may be missing!")
                
            if interp:
                image_features = self.interpolate(image_features)
            if image_features.dim() == 2:
                image_features = image_features.unsqueeze(1) # [B, C] → [B, 1, C]
            return image_features

    # compute the number of patches based on image size and patch size
    def get_output_grid_shape(self, image_size: int = 384):
        dummy = torch.randn(1, 3, image_size, image_size).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output = self._forward(dummy)
            seq_len = output.shape[1]
            grid_size = int(seq_len**0.5)
            return (grid_size, grid_size)

