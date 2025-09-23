from datasets import Dataset, load_from_disk
from detikzify.model import load
from detikzify.train.train import ImageSketchDataset
from detikzify.train.pretrain import tokenize
from PIL import Image
import torch

model, processor = load('detikzify-1B-trained_2048/detikzify-1B')

datikz: Dataset = load_from_disk('datasets/datikz-dataset')
datikz = datikz.select_columns(["image", "code"]).rename_column("code", "text")
eos_token_id, model_max_length = processor.tokenizer.eos_token_id, processor.tokenizer.model_max_length

steps = [
    0,
    512,
    1024, 
    2048, 
    4096
]

for step in range(1, len(steps)):
    dataset = ImageSketchDataset(datikz, processor)
    dataset.filter(
        lambda ex: (ex['input_ids'] == eos_token_id).nonzero() > 0 
        and (ex['input_ids'] == eos_token_id).nonzero() >= steps[step - 1] 
        and (ex['input_ids'] == eos_token_id).nonzero() < steps[step]
    )

    datalist = list(dataset)
    ds = Dataset.from_list(datalist)

    ds.save_to_disk(f"datasets/datikz-filtered_{steps[step]}")