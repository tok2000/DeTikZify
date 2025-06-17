from io import BytesIO
import os
import copy
from random import random
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from ..util import SketchAugment, SplitEpochSaveCallback
from .pretrain import tokenize

logger = logging.get_logger("transformers")

torch.backends.cuda.enable_flash_sdp(False) # added after 17278
torch.backends.cuda.enable_mem_efficient_sdp(False) # added after 17278
torch.backends.cuda.enable_math_sdp(True) # added after 17278

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class ImageSketchDataset(Dataset, TrainerCallback):
    """
    Dataset which samples sketches instead of images, when a sketch exists
    for the current epoch.
    """
    #def __init__(self, dataset, processor, ds_sketch_ratio=.5):
    def __init__(self, dataset, processor, ds_sketch_ratio=0.0):
        super().__init__()
        self.processor = processor
        self.dataset = dataset.with_transform(self.tokenize)
        self.ds_sketch_ratio = ds_sketch_ratio
        self.sketchify = SketchAugment()
        self.cur_epoch = 0

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, batch):
        for idx, sketches in enumerate(batch['image']):
            if isinstance(batch["image"][idx], Image.Image):  # Check if it's a valid image
            	batch["image"][idx] = batch["image"][idx].convert("RGB")  # Ensure RGB format

        return tokenize(
            batch=batch,
            processor=self.processor,
            return_tensors="pt",
            truncation=False,
            padding=True
        )

    def filter(self, *args, **kwargs):
        self.dataset = self.dataset.filter(*args, **kwargs)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.dataset[index]

    def __getitems__(self, indices) -> Dict[str, List[torch.Tensor]]:
        return self.dataset[*indices]

    def on_epoch_end(self, *args, **kwargs):
        self.cur_epoch += 1

        
def train(
    output_dir: str,
    model,
    uncompiled_model,
    processor,
    dataset,
    overwrite=False,
    deepspeed=None,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 1,
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    sketch_ratio=.5,
    gradient_checkpointing: bool = False,
):
    assert num_epochs > 0
    gradient_accumulation_steps = batch_size // micro_batch_size
    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    dataset = ImageSketchDataset(dataset, processor, ds_sketch_ratio=sketch_ratio)
    logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
    eos_token_id, model_max_length = processor.tokenizer.eos_token_id, processor.tokenizer.model_max_length
    dataset.filter(lambda ex: (ex['input_ids'] == eos_token_id).nonzero(as_tuple=True)[0].numel() > 0 and (ex['input_ids'] == eos_token_id).nonzero(as_tuple=True)[0][0].item() < model_max_length) # changed after 16967
    logger.info(f"Dataset size after filtering out too long examples: {len(dataset)}")

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

    trainer = Trainer(
        model=uncompiled_model, # changed after 17229
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            # https://github.com/huggingface/transformers/issues/32576
            gradient_checkpointing_kwargs={'use_reentrant':False},
            warmup_ratio=0.03,
            weight_decay=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            torch_compile=True,
            bf16=True,
            tf32=True,
            logging_steps=10,
            lr_scheduler_type="cosine",
            optim="adamw_torch" if deepspeed else "adamw_torch_fused",
            ddp_find_unused_parameters=True, # changed after 19519
            remove_unused_columns=False,
            save_strategy="epoch",
            report_to="none",
            save_total_limit=1,
            output_dir=output_dir,
            deepspeed=deepspeed,
        ),
        callbacks=[SplitEpochSaveCallback(step_size=0.25)],
        data_collator=lambda batch: batch
    )

    trainer.add_callback(trainer.train_dataset)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if trainer.is_deepspeed_enabled:
        # https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#saving-and-loading
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    #trainer.save_model(output_dir)

    # modified after _orig_mod problem
    model_to_save = uncompiled_model # changed back after 21592
    if hasattr(model_to_save, '_orig_mod'):
        model_to_save = model_to_save._orig_mod
    model_to_save.save_pretrained(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)
    
    trainer.save_state()

    return model, processor