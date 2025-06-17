#!/usr/bin/env -S torchrun --nproc_per_node gpu

import torch._dynamo
torch._dynamo.config.optimize_ddp = False

from argparse import ArgumentParser
from os.path import basename, join, exists

from datasets import Dataset, load_from_disk
from transformers import set_seed
from transformers.utils.logging import enable_explicit_format, set_verbosity_info

from detikzify.dataset import load_dataset
from detikzify.model import load
from detikzify.train import curriculum_train

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune DeTikZify on DaTikZ."
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
    argument_parser.add_argument("--datikz0",
        help="path to the DaTikZ train split processed by the ./sketchify script (in parquet format)",
    )
    argument_parser.add_argument("--datikz1",
        help="path to the DaTikZ train split processed by the ./sketchify script (in parquet format)",
    )
    argument_parser.add_argument("--datikz2",
        help="path to the DaTikZ train split processed by the ./sketchify script (in parquet format)",
    )
    argument_parser.add_argument("--datikz3",
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
    argument_parser.add_argument("--freeze_vision_encoder",
        action="store_true",
        help="possibility to freeze the vision encoder",
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
    model = torch.compile(model)

    datikz_list = [args.datikz0, args.datikz1, args.datikz2, args.datikz3, args.datikz]
    dataset_list = []
    for datikz_elem in datikz_list:
        if datikz_elem is not None:
            datikz: Dataset = load_from_disk(datikz_elem)
            try:
                datikz = datikz.select_columns(["image", "code"]).rename_column("code", "text")
            except:
                datikz = datikz.select_columns(["image", "text"])
            dataset_list.append(datikz)

    curriculum_train(
        model=model,
        uncompiled_model=uncompiled_model, # modified after _orig_mod problem
        processor=processor,
        dataset_list=dataset_list,
        sketch_ratio=0.0,
        output_dir=join(args.output, basename(model.config.name_or_path)), # type: ignore
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        freeze_vision_enc=args.freeze_vision_encoder,
    )
