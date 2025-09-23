from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from detikzify.model import DetikzifyProcessor, DetikzifyForConditionalGeneration, DetikzifyConfig

AutoConfig.register("detikzify", DetikzifyConfig)

model = DetikzifyForConditionalGeneration.from_pretrained("detikzify-1B-trained_1024/detikzify-1B")
processor = DetikzifyProcessor.from_pretrained("detikzify-1B-trained_1024/detikzify-1B")
tokenizer = AutoTokenizer.from_pretrained("detikzify-1B-trained_1024/detikzify-1B")

repo_id = "tok2000/detikzify-1B-trained_1024"

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
processor.push_to_hub(repo_id)

print("Model and tokenizer uploaded successfully!")
