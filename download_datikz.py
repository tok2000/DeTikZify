from datasets import load_dataset

dataset = load_dataset("nllg/datikz-v3", split="train")
dataset.save_to_disk("datasets/datikz-v3")
