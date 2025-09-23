from datasets import Dataset, load_from_disk, load_dataset
from detikzify.model import load
from detikzify.train.train import ImageSketchDataset
from detikzify.train.pretrain import tokenize
from PIL import Image
import torch
from io import BytesIO

from detikzify.util import batchify, convert, expand
from os import sched_getaffinity
from itertools import chain

@batchify
def process_arxivcap(batch, size):
    """Concatenate captions and OCR tokens."""
    for caption_images in chain.from_iterable(batch['caption_images']):
        caption = caption_images['caption']
        for cil_pair in caption_images['cil_pairs']:
            sub_caption = cil_pair['sub_caption']
            text = " ".join(filter(None, [caption, sub_caption]))
            if text:
                yield dict(
                    text=text,
                    image=convert(expand(cil_pair['image'], size, do_trim=True), "png")
                )


model, processor = load('detikzify-1B-trained_2048/detikzify-1B')

ds_stream : Dataset = load_dataset("MMInstruction/ArxivCap", split="train", streaming=True)
arxivcap = ds_stream.shuffle(seed=42)
arxivcap = arxivcap.map(
            process_arxivcap,
            batched=True,
            remove_columns=arxivcap.column_names,
            batch_size=100,
            fn_kwargs=dict(size=model.model.vision_model.config.image_size),
        )
arxivcap = arxivcap.take(500)
arxivlist = list(arxivcap)
sample_dataset = Dataset.from_list(arxivlist)

print(sample_dataset.column_names)

sample_dataset.save_to_disk("datasets/arxivcap-sampled500")