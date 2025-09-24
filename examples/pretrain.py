#!/usr/bin/env -S torchrun --nproc_per_node gpu

from argparse import ArgumentParser
from functools import partial
from itertools import chain
from os.path import basename, join
import torch
import torch

from datasets import Dataset, IterableDataset
from transformers import set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.dataset import load_dataset
from detikzify.model import load
from detikzify.train import pretrain
from detikzify.util import convert, expand, batchify

@batchify
def preprocess(batch, size):
    """Concatenate captions and OCR tokens."""
    for caption_images in chain.from_iterable(batch['caption_images']):
        caption = caption_images['caption']
        for cil_pair in caption_images['cil_pairs']:
            sub_caption = cil_pair['sub_caption']
            ocr = " ".join(cil_pair['image_ocr'])
            if text:=" ".join(filter(None, [caption, sub_caption, ocr])):
                yield dict(
                    text=text,
                    image=convert(expand(cil_pair['image'], size, do_trim=True), "png")
                )





def parse_args():
    argument_parser = ArgumentParser(
        description="Pretrain projection layer of DeTikZify."
    )
    argument_parser.add_argument("--base_model",
        required=True,
        help="The model checkpoint for weights initialization."
    )
    argument_parser.add_argument("--size",
        default=1_000_000,
        type=int,
        help="the amount of figures to use for pretraining"
    )
    argument_parser.add_argument("--output",
        required=True,
        help="directory where to write the model files",
    )
    argument_parser.add_argument("--deepspeed",
        help="path to a DeepSpeed json config file",
    )
    argument_parser.add_argument("--gradient_checkpointing",
        action="store_true",
        help="use gradient checkpointing",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    set_seed(0)

    args = parse_args()
    model, processor = load(args.base_model, ignore_mismatched_sizes=True)
    
    arxivcap: IterableDataset = load_dataset("MMInstruction/ArxivCap", split="train", streaming=True) # type: ignore
    arxivcap = arxivcap.shuffle(0).map(
        preprocess,
        batched=True,
        remove_columns=arxivcap.column_names,
        fn_kwargs=dict(size=model.config.vision_config.image_size),
    )

    import torch._dynamo
    torch._dynamo.config.optimize_ddp = False  # Disable DDP optimization

    # Check if tokens are out-of-range before training
    from torch.utils.data import DataLoader
    
    def custom_collate_fn(batch):
        """Ensures that PIL images are kept as they are, but text is tokenized."""
        processed_batch = {"image": [], "text": []}
    
        for item in batch:
            processed_batch["image"].append(item["image"])  # Keep PIL images as is
            processed_batch["text"].append(item["text"])  # Collect text
    
        return processed_batch

    dataloader = DataLoader(
        Dataset.from_generator(
            generator=partial(iter, arxivcap.take(args.size)),
            features=arxivcap.features,
        ),
        batch_size=4,
        collate_fn=custom_collate_fn  # Use the custom function
    )
    
    pretrain(
        model=model,
        processor=processor,
        output_dir=join(args.output, basename(model.config.name_or_path)),
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        dataset=Dataset.from_generator(
            generator=partial(iter, arxivcap.take(args.size)),
            features=arxivcap.features,
        ),
        batch_size=4  # Reduce this if needed
    )
