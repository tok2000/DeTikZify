import torch
from transformers import AutoTokenizer

@torch.no_grad()
def run_ablation_tests(model, processor, images, prompts, tokenizer=None):
    model.eval()
    print("\nRunning Diagnostic Evaluation...\n")

    tokenizer = tokenizer or processor.tokenizer
    results = {}

    # Variants: full, SVA off, tower1 only, tower2 only
    variants = {
        "full": (images, prompts),
        #"no_images": [],
        #"tower1_only": ([images[0]], [prompts[0]]),
        #"tower2_only": ([images[1]], [prompts[1]]),
    }

    for name, (variant_images, variant_prompts) in variants.items():
        print(f"\nVariant: {name}")
        encoded = processor(images=variant_images, text=variant_prompts, return_tensors="pt")
        print(encoded)
        encoded = {k: v for k, v in encoded.items()}
        encoded.pop("image_hidden_states", None)  # if your model doesn't need it
        print(f"Keys passed to model [{name}]: {list(encoded.keys())}")
        output_ids = model.generate(**encoded, max_new_tokens=20)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Output: {text}")
        results[name] = text

    return results
