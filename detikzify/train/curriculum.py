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

from ..util import SketchAugment, SplitEpochSaveCallback
from .pretrain import tokenize
from .train import ImageSketchDataset

logger = logging.get_logger("transformers")

torch.backends.cuda.enable_flash_sdp(False) # added after 17278
torch.backends.cuda.enable_mem_efficient_sdp(False) # added after 17278
torch.backends.cuda.enable_math_sdp(True) # added after 17278

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class MultiEpochDataset(ImageSketchDataset, TrainerCallback):
    def __init__(self, ds_list: List[Dataset], processor, ds_sketch_ratio=0.0):
        super().__init__()
        self.processor = processor
        self.ds_list = [ds.with_transform(self.tokenize) for ds in dataset]
        self.ds_sketch_ratio = ds_sketch_ratio
        self.sketchify = SketchAugment()
        self.cur_epoch = 0

    def __len__(self):
        return len(self.ds_list[self.cur_epoch])

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self.ds_list[self.cur_epoch][idx]

    def __getitems__(self, indices) -> Dict[str, List[torch.Tensor]]:
        return self.ds_list[self.cur_epoch][*indices]

    def on_epoch_end(self, *args, **kwargs):
        print(f"End of epoch {self.cur_epoch}, dataset length is {len(self)}")
        self.cur_epoch = min(self.cur_epoch + 1, len(self.ds_list) - 1)

def curriculum_train(
    output_dir: str,
    model,
    uncompiled_model,
    processor,
    dataset_list,
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

    dataset = MultiEpochDataset(dataset, processor, ds_sketch_ratio=sketch_ratio)

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

    if freeze_vision_enc:
        logger.info("The vision encoder was frozen. To unfreeze, please start training with --freeze_vision_encoder=False.")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = False
    
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
        callbacks=[SplitEpochSaveCallback(step_size=0.25), dataset],
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