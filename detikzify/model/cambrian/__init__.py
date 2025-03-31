from datasets import DownloadManager
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
    is_timm_available,
)
from transformers.utils.hub import is_remote_url

from .configuration_detikzify import *
from .modeling_detikzify import *
from .processing_detikzify import *
from .sva_module import *

if is_timm_available():
    from detikzify.model.v1 import models as v1_models, load as load_v1

def register():
    try:
        AutoConfig.register("detikzify", DetikzifyConfig)
        AutoModelForVision2Seq.register(DetikzifyConfig, DetikzifyForConditionalGeneration)
        AutoProcessor.register(DetikzifyConfig, DetikzifyProcessor)
    except ValueError:
        pass # already registered

def load(model_name_or_path, modality_projector=None, is_v1=False, **kwargs):
    # backwards compatibility with v1 models
    if is_timm_available() and (is_v1 or model_name_or_path in v1_models): # type: ignore
        model, tokenizer, image_processor = load_v1( # type: ignore
            model_name_or_path=model_name_or_path,
            modality_projector=modality_projector,
            **kwargs
        )
        return model, DetikzifyProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_seq_len=model.config.num_patches,
            image_token=tokenizer.convert_ids_to_tokens(model.config.patch_token_id)
        )

    register()
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)

    # Check if model has multiple encoders
    if len(model.config.mm_vision_tower_aux_list) > 1:
        model.sva = SpatialVisionAggregator(
            q_dim=model.config.text_config.hidden_size,
            kv_dim_list=[
                1024 if "siglip" in enc
                else 1152 if "dino" in enc
                else 1536 if "CLIP-convnext_l" in enc
                else 3072 if "CLIP-convnext_xxl" in enc
                else 1152
                for enc in model.config.mm_vision_tower_aux_list
            ], 
            hidden_dim=model.config.text_config.hidden_size,
            num_heads=model.config.text_config.num_attention_heads,
            num_layers=model.config.sva_layers
        )
    else:
        model.sva = None  # Disable SVA for single encoder setups

    if modality_projector is not None and model.sva is None: # Load modality projector only if SVA is not used
        if is_remote_url(modality_projector):
            modality_projector = DownloadManager().download(modality_projector)
        model.load_state_dict(
            state_dict=load_file(
                filename=modality_projector, # type: ignore
                device=str(model.device)
            ),
            strict=False
        )

    return model, processor
