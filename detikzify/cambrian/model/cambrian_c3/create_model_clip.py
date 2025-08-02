from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoImageProcessor, AutoConfig
)
from detikzify.cambrian.model.cambrian_c3 import (
    DetikzifyCambrianConfig, DetikzifyCambrianForConditionalGeneration, DetikzifyCambrianProcessor
)
from detikzify.cambrian.model.cambrian_c3.encoder import ClipVisionTower, SiglipVisionTower, DinoVisionTower, CLIPConvNextTower
from detikzify.cambrian.model.cambrian_c3 import load
import torch

from types import SimpleNamespace
from PIL import Image

import sys
sys.stdout.reconfigure(encoding='utf-8')

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Load the text model (LLaMA)
print("Loading LLaMA model...")
text_model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
text_model_lm_head = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").lm_head
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 🔹 Define the list of vision encoders
vision_encoders = ["openai/clip-vit-large-patch14-336", "google/siglip-so400m-patch14-384", "facebook/dinov2-base"]
MODEL_MAX_LENGTH = 2048

# 🔹 Load vision encoders once (to be shared between model and processor)
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

# 🔹 Load Detikzify model configuration
print("Loading Detikzify model configuration...")
llama_config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
config = DetikzifyCambrianConfig(
    text_config=llama_config,
    vision_config={"vision_towers": vision_encoders}
)
print(f"After config load: vision_towers={config.vision_towers}")

torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# 🔹 Initialize Detikzify model with pre-loaded encoders
print("Initializing Detikzify model...")
detikzify_model = DetikzifyCambrianForConditionalGeneration(config, preloaded_vision_encoders=loaded_vision_encoders)

# 🔹 Load Detikzify processor with pre-loaded encoders
detikzify_processor = DetikzifyCambrianProcessor(
    image_processor=None,  # Will be set internally by the processor
    tokenizer=tokenizer,
    image_token="<|reserved_special_token_2|>",
    image_seq_len=detikzify_model.model.image_seq_len, # changed after sanity check
    mm_vision_tower_aux_list=config.vision_towers,
    vision_encoders=loaded_vision_encoders,  # Pass pre-loaded encoders
    original_tower_names=vision_encoders  # Pass original clean names
)

# 🔹 Assign text model
detikzify_model.model.text_model.load_state_dict(text_model.state_dict(), strict=False)
detikzify_model.lm_head = text_model_lm_head

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
tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
tokenizer.pad_token_id = VALID_PAD_ID
tokenizer.model_max_length = MODEL_MAX_LENGTH
detikzify_model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model config reflects this

# 🔹 Assign tokenizer and image processor
#detikzify_processor.tokenizer = tokenizer
#detikzify_processor.image_processor = image_processors

# 🔹 Save the model and processor
detikzify_model.save_pretrained(save_directory="detikzify-cambrian-concat-1B-clip_siglip_dino")
detikzify_processor.save_pretrained(save_directory="detikzify-cambrian-concat-1B-clip_siglip_dino")
config.save_pretrained(save_directory="detikzify-cambrian-concat-1B-clip_siglip_dino")
tokenizer.save_pretrained(save_directory="detikzify-cambrian-concat-1B-clip_siglip_dino")


img1 = Image.open("DFA_example_multiplies_of_3.svg.png").convert("RGB")
img2 = Image.open("example2.jpg").convert("RGB")

batch = detikzify_processor(
    images=[img1, img2],
    return_tensors="pt",
    padding=True,
)

input_ids      = batch["input_ids"].to(device)
image_hidden_states = [t.to(device) for t in batch["image_hidden_states"]]

with torch.no_grad():
    outputs = detikzify_model(
        input_ids=input_ids,
        image_hidden_states=image_hidden_states,
        return_dict=True,
    )

print("image_hidden_states shapes:", [t.shape for t in batch["image_hidden_states"]])
print("model.image_hidden_states.shape:", outputs.image_hidden_states.shape)
print("logits.shape:", outputs.logits.shape)
print("# of vision towers:", len(detikzify_model.model.vision_models))
