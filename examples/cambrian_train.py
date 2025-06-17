#!/usr/bin/env -S torchrun --nproc_per_node gpu

import torch._dynamo
torch._dynamo.config.verbose = False # after 21443
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.suppress_errors = True # after 16982
torch._dynamo.disable() # after 21443

from argparse import ArgumentParser
from os.path import basename, join, exists

from datasets import Dataset, load_from_disk
from transformers import set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.cambrian.dataset import load_dataset
from detikzify.cambrian.model.cambrian_v3 import load
from detikzify.cambrian.train import train

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune the DeTikZify-Cambrian model on DaTikZ."
    )
    argument_parser.add_argument("--base_model",
        required=True,
        help="The model checkpoint for weights initialization."
    )
    argument_parser.add_argument(
        "--projector",
        help="url or path to a pretrained modality projector"
    )
    argument_parser.add_argument("--datikz",
        required=True,
        help="path to the DaTikZ train split processed by the ./sketchify script (in parquet format)",
    )
    argument_parser.add_argument("--sketch_ratio",
        default=.5,
        help="ratio of synthetic sketches generated through the ./sketchify script or image transforms",
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
    model, processor = load(args.base_model, modality_projector=args.projector, ignore_mismatched_sizes=True)
    
    import torch
    model.gradient_checkpointing_enable()

    uncompiled_model = model # modified after _orig_mod problem
    model = torch.compile(model, backend="inductor") # changed after 17179, again after 21508, again after 21516, deleted after 21525, added after 21528
    model.config.use_flash_attention = False # added after 17267

    datikz: Dataset = load_from_disk(args.datikz) # type: ignore
    datikz = datikz.select_columns(["image", "code"]).rename_column("code", "text")

    train(
        model=model,
        uncompiled_model=uncompiled_model, # modified after _orig_mod problem
        processor=processor,
        dataset=datikz,
        sketch_ratio=args.sketch_ratio,
        output_dir=join(args.output, basename(model.config.name_or_path)), # type: ignore
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed
    )
