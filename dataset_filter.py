from datasets import Dataset, load_from_disk
from detikzify.model import load
from detikzify.train.train import ImageSketchDataset
from detikzify.train.pretrain import tokenize
from PIL import Image
import torch

class FilteredDataset(ImageSketchDataset):
    def __init__(self, dataset, processor):
        super().__init__(dataset, processor)

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, batch):
        for idx, img in enumerate(batch["image"]):
            if isinstance(img, Image.Image):
                batch["image"][idx] = img.convert("RGB")

        tokenized = tokenize(
            batch=batch,
            processor=self.processor,
            return_tensors="pt",
            truncation=False,
            padding=True
        )

        tokenized["origin"] = batch["origin"]
        tokenized["text"]   = batch["text"]

        return tokenized


model, processor = load('detikzify-1B-trained_2048/detikzify-1B')

datikz: Dataset = load_from_disk('datasets/datikz-dataset')
datikz = datikz.select_columns(["image", "code", "origin"]).rename_column("code", "text")
eos_token_id, model_max_length = processor.tokenizer.eos_token_id, processor.tokenizer.model_max_length
maximum_test = 4096
dataset = FilteredDataset(datikz, processor)
dataset.filter(lambda ex: (ex['input_ids'] == eos_token_id).nonzero() < model_max_length)

datalist = list(dataset)
ds = Dataset.from_list(datalist)

ds.save_to_disk("datasets/datikz-filtered")