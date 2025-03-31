from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoImageProcessor, AutoConfig
)
from detikzify.model.cambrian import (
    DetikzifyCambrianConfig, DetikzifyCambrianForConditionalGeneration, DetikzifyCambrianProcessor,
    SiglipVisionTower, DinoVisionTower, CLIPConvNextTower, SpatialVisionAggregator
)
import torch

from types import SimpleNamespace
from PIL import Image

import sys
sys.stdout.reconfigure(encoding='utf-8')

# 🔹 Load the text model (LLaMA)
print("🔄 Loading LLaMA model...")
text_model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
text_model_lm_head = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").lm_head
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 🔹 Define the list of vision encoders
vision_encoders = ["google/siglip-so400m-patch14-384", "facebook/dinov2-base"]
kv_dim_list=[
    1024 if "siglip" in enc
    else 1152 if "dino" in enc
    else 1536 if "CLIP-convnext-L" in enc
    else 3072 if "CLIP-convnext-XXL" in enc
    else 1152
    for enc in vision_encoders
]

# 🔹 Load vision models dynamically
print(f"🔄 Loading vision encoders: {vision_encoders}")
vision_models = []
image_processors = []

model_map = {
    "CLIP-convnext-L": "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
    "CLIP-convnext-XXL": "laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup"
}

for vision_encoder in vision_encoders:
    if "siglip" in vision_encoder:
        print(f"🚀 Loading Siglip model: {vision_encoder}")
        vision_model = SiglipVisionTower(vision_encoder, args={})
    elif "dino" in vision_encoder:
        print(f"🚀 Loading Dino model: {vision_encoder}")
        vision_model = DinoVisionTower(vision_encoder, args={})
    elif "CLIP-convnext" in vision_encoder:
        print(f"🚀 Loading Clip-ConvNext model: {vision_encoder}")
        vision_model = CLIPConvNextTower(vision_encoder, args=SimpleNamespace(mm_vision_select_layer=12, mm_vision_select_feature="patch", unfreeze_mm_vision_tower=False), delay_load=True)
        vision_model.load_model()
    else:
        raise ValueError(f"❌ Unsupported vision encoder: {vision_encoder}")

    vision_models.append(vision_model)
    
    hf_model_name = model_map.get(vision_encoder, vision_encoder)
    if "CLIP-convnext" in vision_encoder:
        image_processors.append(vision_model.image_processor)  # Already wrapped
    else:
        image_processors.append(AutoImageProcessor.from_pretrained(hf_model_name))

# 🔹 Load Detikzify model configuration
print("🔄 Loading Detikzify model configuration...")
llama_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
config = DetikzifyCambrianConfig(
    text_config=llama_config,
    mm_vision_tower_aux_list=vision_encoders,
    query_num_list=[8, 8]
)
print(f"✅ After config load: mm_vision_tower_aux_list={config.mm_vision_tower_aux_list}")

# 🔹 Load Detikzify processor
detikzify_processor = DetikzifyCambrianProcessor.from_pretrained("nllg/detikzify-v2-8b")

# 🔹 Initialize Detikzify model
print("🔄 Initializing Detikzify model...")
detikzify_model = DetikzifyCambrianForConditionalGeneration(config)

# 🔹 Assign text model
detikzify_model.model.text_model.load_state_dict(text_model.state_dict(), strict=False)
detikzify_model.lm_head = text_model_lm_head

# 🔹 Assign vision models (handle single or multiple models)
if len(vision_models) > 1:
    detikzify_model.model.vision_models = torch.nn.ModuleList(vision_models)
else:
    detikzify_model.model.vision_model = vision_models[0]

detikzify_model.model.sva = SpatialVisionAggregator(
    q_dim=2048,
    kv_dim_list=kv_dim_list, 
    hidden_dim=2048,
    num_heads=detikzify_model.model.config.text_config.num_attention_heads,
    num_layers=detikzify_model.model.config.sva_layers
)

# 🔹 Update config with text model details
for key, value in vars(text_model.config).items():
    setattr(detikzify_model.config.text_config, key, value)

# 🔹 Ensure correct `pad_token_id` before assigning
VALID_PAD_ID = 128004  # Set to an existing valid ID

# 🔹 Step 1: Remove `[PAD]` if it's assigned incorrectly at 128256
if tokenizer.pad_token_id == 128256:
    print(f"Removing previous PAD token at ID {tokenizer.pad_token_id}")
    del tokenizer.special_tokens_map['pad_token']  # Remove from special tokens
    tokenizer.pad_token_id = None  # Unset pad token

# 🔹 Step 2: Explicitly set `[PAD]` to `128004`
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = VALID_PAD_ID
detikzify_model.config.pad_token_id = VALID_PAD_ID  # Ensure model config reflects this

# 🔹 Assign tokenizer and image processor
detikzify_processor.tokenizer = tokenizer
detikzify_processor.image_processor = image_processors[0]  # Use first image processor

# 🔹 Save the model and processor
detikzify_model.save_pretrained(save_directory="detikzify-cambrian-1B-siglip_dino")
detikzify_processor.save_pretrained(save_directory="detikzify-cambrian-1B-siglip_dino")
tokenizer.save_pretrained("detikzify-cambrian-1B-siglip_dino")

# 🔹 Print loaded vision models
#print("\n✅ Loaded Vision Models:")
#for i, model in enumerate(detikzify_model.model.vision_models):
#    print(f"   🔹 Vision Model {i+1}: {model}")

# 🔹 Run a test inference to verify both encoders work
print("\n🖼️ Running vision model test...")
dummy_vision_input = torch.randn(1, 3, 384, 384).to(detikzify_model.device)  # Adjust size if needed

# 🔹 Ensure both vision models process input correctly
queries = torch.randn(1, 4, 2048).to(detikzify_model.device)  # Ensure correct query shape
vision_latents_list = [torch.randn(1, 729, dim) for dim in kv_dim_list]  # Ensure vision latents match kv_dim_list

print("\n🚀 Checking vision latents sent to SVA:")
for i, latents in enumerate(vision_latents_list):
    print(f"   🔹 Vision Model {i+1} Latents Shape: {latents.shape}")

# 🔹 Pass latents to SVA module
aggregated_features = detikzify_model.model.sva(queries, vision_latents_list)

print(f"\n✅ SVA Output Shape: {aggregated_features.shape}")

# 🔹 Generate text using the Detikzify model
print("\n📝 Running text generation test...")
text = "The key to life is"
inputs = tokenizer(text, return_tensors="pt").to(detikzify_model.device)

output = detikzify_model.generate(**inputs)
print("\n✨ Generated Output:")
print(tokenizer.decode(output[0], skip_special_tokens=True))

print("\n📸 Running image processing test...")

# 🔹 Load test image
image = Image.open("DFA_example_multiplies_of_3.svg.png").convert("RGB")

# 🔹 Preprocess with your Detikzify processor
processed = detikzify_processor(
    text="Describe this image:",
    images=image,
    return_tensors="pt"
).to(detikzify_model.device)

# 🔹 Forward pass (no generation yet)
with torch.no_grad():
    output = detikzify_model(
        input_ids=processed["input_ids"],
        attention_mask=processed["attention_mask"],
        pixel_values=processed["pixel_values"],
        return_dict=True
    )

# 🔍 Check the shape of image hidden states
print("📸 image_hidden_states:", output.image_hidden_states.shape if output.image_hidden_states is not None else "None")

#print("\n🛠️ Model Configuration:")
#print(detikzify_model.config)
