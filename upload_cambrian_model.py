import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from detikzify.cambrian.model.cambrian_c3 import DetikzifyCambrianProcessor, DetikzifyCambrianForConditionalGeneration, DetikzifyCambrianConfig

def main():
    parser = argparse.ArgumentParser(description="Upload model to HF Hub")
    parser.add_argument("--model_path", required=True, help="Path to local model folder")
    parser.add_argument("--hf_repo_id", required=True, help="Hugging Face model repo ID")
    args = parser.parse_args()

    model_path = args.model_path
    repo_id = args.hf_repo_id

    model = DetikzifyCambrianForConditionalGeneration.from_pretrained(model_path)
    processor = DetikzifyCambrianProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)

    print("Model and tokenizer uploaded successfully!")

if __name__ == "__main__":
    main()