from argparse import ArgumentParser
from os.path import basename, join, exists
from transformers.utils.logging import enable_explicit_format, set_verbosity_info
from transformers import set_seed
from datasets import Dataset, load_from_disk, load_dataset

from detikzify.rl.model import load

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune the DeTikZify-Cambrian model on DaTikZ using Reinforcement Learning."
    )
    argument_parser.add_argument("--base_model",
        required=True,
        help="The model checkpoint for weights initialization."
    )
    argument_parser.add_argument("--datikz",
        required=True,
        help="path to the DaTikZ train split processed by the ./sketchify script (in parquet format)",
    )
    argument_parser.add_argument("--output",
        required=True,
        help="directory where to write the model files",
    )
    argument_parser.add_argument("--rl_type",
        default="grpo",
        help="The model checkpoint for weights initialization."
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
    model, processor = load(args.base_model, ignore_mismatched_sizes=True)

    model.gradient_checkpointing_enable()
    model.config.use_flash_attention = False

    dataset: Dataset = load_from_disk(args.datikz)
    dataset = dataset.select_columns(["image", "code"]).rename_column("code", "text")

    rl_type = args.rl_type.lower()
    if rl_type == "grpo":
        from detikzify.rl.train.train_grpo import train
    elif rl_type == "ppo":
        from detikzify.rl.train.train_ppo import train
    else:
        raise ValueError(
            f"The reinforcement learning method ({rl_type}) is not implemented yet. "
            "Use `grpo` or `ppo` to overcome."
        )

    train(
        model=model,
        processor=processor,
        dataset=dataset,
        output_dir=join(args.output, basename(model.config.name_or_path)),
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        freeze_vision_enc=args.freeze_vision_encoder,
    )