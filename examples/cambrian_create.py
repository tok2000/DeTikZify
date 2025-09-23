from argparse import ArgumentParser

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoImageProcessor, AutoConfig
)
from detikzify.cambrian.model.cambrian_concat import (
    DetikzifyCambrianConfig, DetikzifyCambrianForConditionalGeneration, DetikzifyCambrianProcessor
)
from detikzify.cambrian.model.cambrian_concat.encoder import ClipVisionTower, SiglipVisionTower, DinoVisionTower, CLIPConvNextTower
from detikzify.cambrian.model.cambrian_concat import load
import torch

from types import SimpleNamespace
from PIL import Image

import sys
sys.stdout.reconfigure(encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    argument_parser = ArgumentParser(
        description="Create a Cambrian model with multiple vision encoders and LLaMA text model."
    )
    argument_parser.add_argument("--vision_encoders",
        required=True,
        nargs='+',
        default=["google/siglip-so400m-patch14-384", "facebook/dinov2-base"],
        help="List of vision encoder model names to include in the Cambrian model.",
    )
    argument_parser.add_argument("--output_dir",
        required=True,
        help="directory where to write the model files",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_seed(0)
    set_verbosity_info()
    enable_explicit_format()

    valid_model_prefixes = {
        "openai/clip-vit-large-patch14-336":"openai/clip-vit-large-patch14-336",
        "clip-vit":"openai/clip-vit-large-patch14-336",
        "clip-vit-large-patch14-336":"openai/clip-vit-large-patch14-336",
        "clip":"openai/clip-vit-large-patch14-336",
        "CLIP-ViT-L/14@336":"openai/clip-vit-large-patch14-336",

        "google/siglip-so400m-patch14-384":"google/siglip-so400m-patch14-384",
        "siglip":"google/siglip-so400m-patch14-384",
        "siglip-so400m-patch14-384":"google/siglip-so400m-patch14-384",
        "SigLIP-ViT-SO400M/14@384":"google/siglip-so400m-patch14-384",

        "facebook/dinov2-base":"facebook/dinov2-base",
        "dino":"facebook/dinov2-base",
        "dinov2-base":"facebook/dinov2-base",
        "dinov2":"facebook/dinov2-base",
        "DINOv2-ViT-L/14@518":"facebook/dinov2-base",

        "CLIP-convnext-L":"CLIP-convnext-L",

        "CLIP-convnext-XXL":"CLIP-convnext-XXL",
        "clip-convnext":"CLIP-convnext-XXL",
        "clip-convnext-XXL":"CLIP-convnext-XXL",
        "clip-convnext-xxl":"CLIP-convnext-XXL",
        "convnext-xxl":"CLIP-convnext-XXL",
        "convnext-xxlarge":"CLIP-convnext-XXL",
    }

    # Load the text model (LLaMA)
    print("Loading LLaMA model...")
    text_model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
    text_model_lm_head = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").lm_head
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Define the list of vision encoders
    for i, ve in enumerate(args.vision_encoders):
        if ve in valid_model_prefixes:
            args.vision_encoders[i] = valid_model_prefixes[ve]
        else:
            raise ValueError(f"Unsupported vision encoder specified: {ve}")
    vision_encoders = args.vision_encoders
    
    MODEL_MAX_LENGTH = 2048

    # Load vision encoders once (to be shared between model and processor)
    print(f"Loading vision encoders: {vision_encoders}")
    loaded_vision_encoders = []

    for vision_encoder in vision_encoders:
        if "clip-vit" in vision_encoder:
            print(f"Loading Clip model: {vision_encoder}")
            encoder = ClipVisionTower(vision_encoder, args={})
        elif "siglip" in vision_encoder:
            print(f"Loading Siglip model: {vision_encoder}")
            encoder = SiglipVisionTower(vision_encoder, args={})
        elif "dino" in vision_encoder:
            print(f"Loading Dino model: {vision_encoder}")
            encoder = DinoVisionTower(vision_encoder, args={})
        elif "CLIP-convnext" in vision_encoder:
            print(f"Loading Clip-ConvNext model: {vision_encoder}")
            encoder = CLIPConvNextTower(vision_encoder, args=SimpleNamespace(mm_vision_select_layer=12, mm_vision_select_feature="patch", unfreeze_mm_vision_tower=False), delay_load=True)
            encoder.load_model()
        else:
            raise ValueError(f"Unsupported vision encoder: {vision_encoder}")

        loaded_vision_encoders.append(encoder)

    # Load Detikzify model configuration
    print("Loading Detikzify model configuration...")
    llama_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    config = DetikzifyCambrianConfig(
        text_config=llama_config,
        vision_config={"vision_towers": vision_encoders}
    )
    print(f"After config load: vision_towers={config.vision_towers}")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Initialize Detikzify model with pre-loaded encoders
    print("Initializing Detikzify model...")
    detikzify_model = DetikzifyCambrianForConditionalGeneration(config, preloaded_vision_encoders=loaded_vision_encoders)

    # Load Detikzify processor with pre-loaded encoders
    detikzify_processor = DetikzifyCambrianProcessor(
        image_processor=None, # Will be set internally by the processor
        tokenizer=tokenizer,
        image_token="<|reserved_special_token_2|>",
        image_seq_len=detikzify_model.model.image_seq_len,
        mm_vision_tower_aux_list=config.vision_towers,
        vision_encoders=loaded_vision_encoders, # Pass pre-loaded encoders
        original_tower_names=vision_encoders # Pass original clean names
    )

    # Assign text model
    detikzify_model.model.text_model.load_state_dict(text_model.state_dict(), strict=False)
    detikzify_model.lm_head = text_model_lm_head

    # Update config with text model details
    for key, value in vars(text_model.config).items():
        setattr(detikzify_model.config.text_config, key, value)

    VALID_PAD_ID = 128004

    # Step 1: Remove `[PAD]` if it's assigned incorrectly at 128256
    if tokenizer.pad_token_id == 128256:
        print(f"Removing previous PAD token at ID {tokenizer.pad_token_id}")
        del tokenizer.special_tokens_map['pad_token']  # Remove from special tokens
        tokenizer.pad_token_id = None  # Unset pad token

    # Step 2: Explicitly set `[PAD]` to `128004`
    tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
    tokenizer.pad_token_id = VALID_PAD_ID
    tokenizer.model_max_length = MODEL_MAX_LENGTH
    detikzify_model.config.pad_token_id = tokenizer.pad_token_id

    # Save the model and processor
    detikzify_model.save_pretrained(save_directory=args.output_dir)
    detikzify_processor.save_pretrained(save_directory=args.output_dir)
    config.save_pretrained(save_directory=args.output_dir)
    tokenizer.save_pretrained(save_directory=args.output_dir)
