from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoImageProcessor, AutoConfig
)
from detikzify.cambrian.model.cambrian_sva import (
    DetikzifyCambrianConfig, DetikzifyCambrianForConditionalGeneration, DetikzifyCambrianProcessor, SpatialVisionAggregator
)
from detikzify.cambrian.model.cambrian_sva.encoder import ClipVisionTower, SiglipVisionTower, DinoVisionTower, CLIPConvNextTower
from detikzify.cambrian.model.cambrian_sva.preevaluate import run_ablation_tests
from detikzify.cambrian.model.cambrian_sva import load
import torch

from types import SimpleNamespace
from PIL import Image

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the text model (LLaMA)
print("Loading LLaMA model...")
text_model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
text_model_lm_head = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").lm_head
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Define the list of vision encoders
vision_encoders = ["openai/clip-vit-large-patch14-336", "google/siglip-so400m-patch14-384", "facebook/dinov2-base", "CLIP-convnext-XXL"]
spatial_grid_size = (18, 18)
vision_fusion_layer_stride = 3
vision_fusion_layer_start = 0
num_groups = 1 # G from Cambrian
MODEL_MAX_LENGTH = 2048

num_fusion_layers = text_model.config.num_hidden_layers // vision_fusion_layer_stride
vision_fusion_layers = [vision_fusion_layer_start + vision_fusion_layer_stride * i for i in range(num_fusion_layers)]

# Load vision models dynamically
print(f"Loading vision encoders: {vision_encoders}")
vision_models = []
image_processors = []
hf_model_names = []

model_map = {
    "CLIP-convnext-L": "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
    "CLIP-convnext-XXL": "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
}

for vision_encoder in vision_encoders:
    if "clip-vit" in vision_encoder:
        print(f"Loading Clip model: {vision_encoder}")
        vision_model = ClipVisionTower(vision_encoder, args={})
    elif "siglip" in vision_encoder:
        print(f"Loading Siglip model: {vision_encoder}")
        vision_model = SiglipVisionTower(vision_encoder, args={})
    elif "dino" in vision_encoder:
        print(f"Loading Dino model: {vision_encoder}")
        vision_model = DinoVisionTower(vision_encoder, args={})
    elif "CLIP-convnext" in vision_encoder:
        print(f"Loading Clip-ConvNext model: {vision_encoder}")
        vision_model = CLIPConvNextTower(vision_encoder, args=SimpleNamespace(mm_vision_select_layer=12, mm_vision_select_feature="patch", unfreeze_mm_vision_tower=False), delay_load=True)
        vision_model.load_model()
    else:
        raise ValueError(f"Unsupported vision encoder: {vision_encoder}")

    vision_models.append(vision_model)
    
    hf_model_name = model_map.get(vision_encoder, vision_encoder)
    hf_model_names.append(hf_model_name)
    if "CLIP-convnext" in vision_encoder:
        image_processors.append(vision_model.image_processor)  # Already wrapped
    else:
        image_processors.append(AutoImageProcessor.from_pretrained(hf_model_name))

with torch.no_grad():
    dummy = torch.randn(1, 3, 384, 384).to("cuda" if torch.cuda.is_available() else "cpu")
    kv_dim_list = [model(dummy).shape[-1] for model in vision_models]

# Load Detikzify model configuration
print("Loading Detikzify model configuration...")
llama_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
config = DetikzifyCambrianConfig(
    text_config=llama_config,
    mm_vision_tower_aux_list=vision_encoders,
    vision_fusion_layers=vision_fusion_layers,
    spatial_grid_size=spatial_grid_size,
    num_groups=num_groups
)
print(f"After config load: mm_vision_tower_aux_list={config.mm_vision_tower_aux_list}")

torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# Initialize Detikzify model
print("Initializing Detikzify model...")
detikzify_model = DetikzifyCambrianForConditionalGeneration(config)

# Load Detikzify processor
image_processor = image_processors[1]  # or combine if needed
detikzify_processor = DetikzifyCambrianProcessor(
    image_processor=image_processors[1],
    tokenizer=tokenizer,
    image_token="<|reserved_special_token_2|>",
    image_seq_len=detikzify_model.model.image_seq_len, # changed after sanity check
    mm_vision_tower_aux_list=config.mm_vision_tower_aux_list
)

# Assign text model
detikzify_model.model.text_model.load_state_dict(text_model.state_dict(), strict=False)
detikzify_model.lm_head = text_model_lm_head

# Assign vision models (handle single or multiple models)
if len(vision_models) > 1:
    detikzify_model.model.vision_models = torch.nn.ModuleList(vision_models)
else:
    detikzify_model.model.vision_model = vision_models[0]

# Update config with text model details
for key, value in vars(text_model.config).items():
    setattr(detikzify_model.config.text_config, key, value)

# Ensure correct `pad_token_id` before assigning
VALID_PAD_ID = 128004  # Set to an existing valid ID

# Step 1: Remove `[PAD]` if it's assigned incorrectly at 128256
if tokenizer.pad_token_id == 128256:
    print(f"Removing previous PAD token at ID {tokenizer.pad_token_id}")
    del tokenizer.special_tokens_map['pad_token']  # Remove from special tokens
    tokenizer.pad_token_id = None  # Unset pad token

# Step 2: Explicitly set `[PAD]` to `128004`
tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
tokenizer.pad_token_id = VALID_PAD_ID
tokenizer.model_max_length = MODEL_MAX_LENGTH
detikzify_model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model config reflects this

# Save the model and processor
detikzify_model.save_pretrained(save_directory="detikzify-cambrian-1B-clip_siglip_dino_convnext-XXL")
detikzify_processor.save_pretrained(save_directory="detikzify-cambrian-1B-clip_siglip_dino_convnext-XXL")
config.save_pretrained(save_directory="detikzify-cambrian-1B-clip_siglip_dino_convnext-XXL")
tokenizer.save_pretrained(save_directory="detikzify-cambrian-1B-clip_siglip_dino_convnext-XXL")

detikzify_model.save_pretrained(save_directory="test_model")
detikzify_processor.save_pretrained(save_directory="test_model")
config.save_pretrained(save_directory="test_model")
tokenizer.save_pretrained(save_directory="test_model")

detikzify_model, detikzify_processor = load("test_model", ignore_mismatched_sizes=True)

# Run a test inference to verify both encoders work
print("\nRunning vision model test...")
dummy_vision_input = torch.randn(1, 3, 384, 384).to(detikzify_model.device)  # Adjust size if needed

# Ensure both vision models process input correctly
#queries = torch.randn(1, config.spatial_grid_size[0] * config.spatial_grid_size[1], 2048).to(detikzify_model.device)  # Ensure correct query shape
queries = torch.randn(1, spatial_grid_size[0] * spatial_grid_size[1], 2048).to(detikzify_model.device)  # Ensure correct query shape
vision_latents_list = [encoder(dummy_vision_input) for encoder in detikzify_model.model.vision_models]  # Ensure vision latents match kv_dim_list

print("\nChecking vision latents sent to SVA:")
for i, latents in enumerate(vision_latents_list):
    print(f"   Vision Model {i+1} Latents Shape: {latents.shape}")
    