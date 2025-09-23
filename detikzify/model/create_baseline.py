from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, SiglipVisionModel, AutoImageProcessor
from detikzify.model import DetikzifyConfig, DetikzifyForConditionalGeneration, DetikzifyProcessor

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the model and tokenizer
text_model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B")
text_model_lm_head = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").lm_head
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

vision_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
image_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Load the Detikzify model
config = DetikzifyConfig.from_pretrained("nllg/detikzify-v2-8b")
detikzify_processor = DetikzifyProcessor.from_pretrained("nllg/detikzify-v2-8b")

detikzify_model = DetikzifyForConditionalGeneration(config)

detikzify_model.model.text_model = text_model
detikzify_model.model.vision_model = vision_model
detikzify_model.lm_head = text_model_lm_head

# Transfer config settings from the base models to the Detikzify model
for key, value in vars(vision_model.config).items():
    setattr(detikzify_model.config.vision_config, key, value)

for key, value in vars(text_model.config).items():
    setattr(detikzify_model.config.text_config, key, value)

detikzify_processor.tokenizer = tokenizer
detikzify_processor.image_processor = image_processor

detikzify_model.save_pretrained(save_directory="detikzify-1B")
detikzify_processor.save_pretrained(save_directory="detikzify-1B")
