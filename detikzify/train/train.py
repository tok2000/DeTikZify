from io import BytesIO
import os
from random import random
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging
import pickle
from datasets import load_from_disk as hf_load_from_disk
from transformers import AutoProcessor

from ..util import SketchAugment, SplitEpochSaveCallback
from .pretrain import tokenize

logger = logging.get_logger("transformers")

# added to disable flash attention, which caused OOMs in some cases
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class ImageSketchDataset(Dataset, TrainerCallback):
    """
    Dataset which samples sketches instead of images, when a sketch exists
    for the current epoch.
    """
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
            if isinstance(batch["image"][idx], Image.Image): # Check if it's a valid image
                batch["image"][idx] = batch["image"][idx].convert("RGB") # Ensure RGB format

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

    # For saving and loading the dataset on disk
    def save_to_disk(self, path):
        os.makedirs(path, exist_ok=True)
        self.dataset.set_transform(None)
        self.dataset.save_to_disk(os.path.join(path, "dataset"))
        self.dataset.set_transform(self.tokenize)
        self.processor.save_pretrained(os.path.join(path, "processor"))
        metadata = {
            "ds_sketch_ratio": self.ds_sketch_ratio,
            "cur_epoch": self.cur_epoch
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

    # Load the dataset and processor from disk
    @classmethod
    def load_from_disk(cls, path, processor=None):
        dataset = hf_load_from_disk(os.path.join(path, "dataset"))
        if processor is None:
            processor = AutoProcessor.from_pretrained(os.path.join(path, "processor"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        instance = cls(dataset, processor, ds_sketch_ratio=metadata["ds_sketch_ratio"])
        instance.cur_epoch = metadata["cur_epoch"]
        return instance
    

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
    freeze_vision_enc: bool = False,
):
    assert num_epochs > 0
    gradient_accumulation_steps = batch_size // micro_batch_size
    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    dataset = ImageSketchDataset(dataset, processor, ds_sketch_ratio=sketch_ratio)
    logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
    eos_token_id, model_max_length = processor.tokenizer.eos_token_id, processor.tokenizer.model_max_length
    dataset.filter(lambda ex: (ex['input_ids'] == eos_token_id).nonzero() < model_max_length)
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

    # Freeze vision encoder if specified
    if freeze_vision_enc:
        logger.info("The vision encoder was frozen. To unfreeze, please start training with --freeze_vision_encoder=False.")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = False
    
    trainer = Trainer(
        model=uncompiled_model, # changed in order to avoid _orig_mod problem
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
            ddp_find_unused_parameters=True,
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

    # modified after _orig_mod problem
    model_to_save = uncompiled_model
    if hasattr(model_to_save, '_orig_mod'):
        model_to_save = model_to_save._orig_mod
    model_to_save.save_pretrained(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)
    
    trainer.save_state()

    return model, processor