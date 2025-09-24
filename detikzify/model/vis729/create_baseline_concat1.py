from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, SiglipVisionModel, AutoImageProcessor
from detikzify.model import DetikzifyConfig, DetikzifyProcessor
from detikzify.model.vis729 import DetikzifyForConditionalGeneration

import sys
sys.stdout.reconfigure(encoding='utf-8')

VALID_PAD_ID = 128004
MODEL_MAX_LENGTH = 2048

# Load the model and tokenizer
text_model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
text_model_lm_head = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").lm_head
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

vision_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
image_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Load the Detikzify model
config = DetikzifyConfig.from_pretrained("nllg/detikzify-v2-8b")

for key, value in vars(vision_model.config).items():
    setattr(config.vision_config, key, value)

for key, value in vars(text_model.config).items():
    setattr(config.text_config, key, value)
    
config.concat_factor = 1

detikzify_model = DetikzifyForConditionalGeneration(config)

detikzify_model.model.text_model = text_model
detikzify_model.model.vision_model = vision_model
detikzify_model.lm_head = text_model_lm_head

detikzify_processor = DetikzifyProcessor(image_processor, tokenizer)
detikzify_processor.image_seq_len = 729

if detikzify_processor.tokenizer.pad_token_id == 128256:
    print(f"Removing previous PAD token at ID {tokenizer.pad_token_id}")
    del detikzify_processor.tokenizer.special_tokens_map['pad_token']
    detikzify_processor.tokenizer.pad_token_id = None

detikzify_processor.tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
detikzify_processor.tokenizer.pad_token_id = VALID_PAD_ID
detikzify_processor.tokenizer.model_max_length = MODEL_MAX_LENGTH

detikzify_model.config.pad_token_id = detikzify_processor.tokenizer.pad_token_id

detikzify_model.save_pretrained(save_directory="/work/tikrause/DeTikZify/detikzify-1B-729vis")
detikzify_processor.save_pretrained(save_directory="/work/tikrause/DeTikZify/detikzify-1B-729vis")

# Generate text using the Detikzify model
text = "The key to life is"

inputs = tokenizer(text, return_tensors="pt")
output = detikzify_model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))

print(detikzify_model.config)