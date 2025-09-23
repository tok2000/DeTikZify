import torch
import torch.nn as nn
from transformers.utils import logging
from transformers.trainer_utils import get_last_checkpoint
import os
import torch.distributed as dist
from argparse import ArgumentParser
from transformers.utils.logging import enable_explicit_format, set_verbosity_info
from transformers import set_seed
from datasets import Dataset, load_from_disk, load_dataset

from detikzify.rl.model import load

from detikzify.rl.infer.tikz import TikzDocument
from detikzify.rl.evaluate.imagesim import ImageSim
from detikzify.rl.util import SplitEpochSaveCallback

from trl import GRPOTrainer, GRPOConfig

logger = logging.get_logger("grpo_trainer")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1)) # to be clarified
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")


class RewardFunction:
    def __init__(self, model, processor):
        self.reward_model = ImageSim.from_detikzify(model, processor, sync_on_compute=False)
        self.reward_model.to(device)
        self.processor = processor
        self.__name__ = "ImageSimRewardFunction"

    def __call__(self, prompts, completions, image: list, **kwargs):
        rewards = []
        for sketch_image, tikz_code in zip(image, completions):
            tikz_doc = TikzDocument(code=tikz_code)
            if tikz_doc.is_rasterizable:
                rendered_image = tikz_doc.rasterize()
                self.reward_model.update(rendered_image, sketch_image)
                reward = self.reward_model.compute()
            else:
                reward = -1.0 # negative reward if doc is not rasterizable
            self.reward_model.reset()
            rewards.append(reward)

        return rewards
    
    
def train(
    output_dir: str,
    model,
    processor,
    dataset,
    overwrite=False,
    deepspeed=None,
    batch_size: int = 8,
    micro_batch_size: int = 1,
    num_train_steps: int = 500,
    learning_rate: float = 1e-6,
    gradient_checkpointing: bool = False,
    freeze_vision_enc: bool = False,
    num_gen: int = 8,
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    eos_token_id, model_max_length = processor.tokenizer.eos_token_id, processor.tokenizer.model_max_length
    
    last_checkpoint = None
    if os.path.isdir(output_dir) and not overwrite:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use `overwrite` to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `output_dir` or add `overwrite` to train from scratch."
            )

    # Freeze vision encoder if specified
    if freeze_vision_enc:
        logger.info("The vision encoder was frozen. To unfreeze, please start training with --freeze_vision_encoder=False.")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = False
    model.enable_input_require_grads()

    reward_fn = RewardFunction(model, processor)

    training_args = GRPOConfig(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps * num_gen,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        weight_decay=0.01,
        epsilon=0.4,
        temperature=max(1., model.generation_config.temperature),
        top_p=model.generation_config.top_p,
        top_k=model.generation_config.top_k,
        max_steps=num_train_steps,
        learning_rate=learning_rate,
        torch_compile=False,
        bf16=not deepspeed,
        fp16=deepspeed,
        tf32=True,
        logging_steps=10,
        lr_scheduler_type="cosine",
        optim="adamw_torch" if deepspeed else "adamw_torch_fused",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        save_total_limit=1,
        output_dir=output_dir,
        deepspeed=deepspeed,
        
        max_completion_length=processor.tokenizer.model_max_length-processor.image_seq_len,
        max_prompt_length=None,
        num_generations=num_gen,
        log_completions=True,
        num_completions_to_print=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        callbacks=[SplitEpochSaveCallback(step_size=0.25)],
    )
    
    import numpy.core.multiarray
    import torch.serialization
    torch.serialization.add_safe_globals({numpy.core.multiarray._reconstruct})

    trainer.generation_config.bad_words_ids = [[model.config.image_token_id]]
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if trainer.is_deepspeed_enabled:
        # https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#saving-and-loading
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    trainer.save_model(output_dir)

    return model, processor

def parse_args():
    argument_parser = ArgumentParser(
        description="Fine-tune the DeTikZify model on DaTikZ using Reinforcement Learning."
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
    argument_parser.add_argument("--batch_size",
        type=int,
        default=8,
        help="batch size for training (default: 8)",
    )
    argument_parser.add_argument("--micro_batch_size",
        type=int,
        default=1,
        help="micro batch size per GPU (default: 1)",
    )
    argument_parser.add_argument("--num_completions",
        type=int,
        default=8,
        help="number of completions to generate for each input (default: 8)",
    )
    argument_parser.add_argument("--num_train_steps",
        default=500,
        type=int,
        help="number of training steps to run GRPO for",
    )

    return argument_parser.parse_args()

if __name__ == "__main__":
    set_verbosity_info()
    enable_explicit_format()
    set_seed(0)

    args = parse_args()
    model, processor = load(args.base_model)

    model.config.use_flash_attention = False

    dataset: Dataset = load_from_disk(args.datikz)
    dataset = dataset.map(lambda x: {"text": ""})
    #dataset = dataset.select_columns(["image", "code"]).rename_column("code", "text")

    train(
        model=model,
        processor=processor,
        dataset=dataset,
        output_dir=os.path.join(args.output, os.path.basename(model.config.name_or_path)),
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        freeze_vision_enc=args.freeze_vision_encoder,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_gen=args.num_completions,
    )