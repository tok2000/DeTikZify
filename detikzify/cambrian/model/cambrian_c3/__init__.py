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

if is_timm_available():
    from detikzify.model.v1 import models as v1_models, load as load_v1

def register():
    try:
        AutoConfig.register("detikzify", DetikzifyCambrianConfig)
        AutoModelForVision2Seq.register(DetikzifyCambrianConfig, DetikzifyCambrianForConditionalGeneration)
        AutoProcessor.register(DetikzifyCambrianConfig, DetikzifyCambrianProcessor)
        print("Successfully registered DetikzifyCambrianProcessor with AutoProcessor")
    except ValueError as e:
        print(f"Registration warning (may already be registered): {e}")
    except Exception as e:
        print(f"Registration error: {e}")
        print("Will use direct processor loading as fallback")

def load(model_name_or_path, modality_projector=None, is_v1=False, **kwargs):
    # backwards compatibility with v1 models
    if is_timm_available() and (is_v1 or model_name_or_path in v1_models): # type: ignore
        model, tokenizer, image_processor = load_v1( # type: ignore
            model_name_or_path=model_name_or_path,
            modality_projector=modality_projector,
            **kwargs
        )
        return model, DetikzifyCambrianProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_seq_len=model.config.num_patches,
            image_token=tokenizer.convert_ids_to_tokens(model.config.patch_token_id)
        )

    register()
    
    # Try to load processor with AutoProcessor first
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        # Check if we got the right type of processor
        if not hasattr(processor, 'image_token'):
            print(f"Warning: AutoProcessor returned {type(processor)}, expected DetikzifyCambrianProcessor")
            print("Falling back to direct processor loading...")
            processor = DetikzifyCambrianProcessor.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"AutoProcessor failed: {e}")
        print("Falling back to direct processor loading...")
        processor = DetikzifyCambrianProcessor.from_pretrained(model_name_or_path)
    
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    
    if modality_projector is not None:
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
    