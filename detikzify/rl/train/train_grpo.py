import torch
import torch.nn as nn
from copy import deepcopy
from datasets import Dataset, load_from_disk
from trl import GRPOConfig
from transformers.utils import logging
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
import os

from .train import ImageSketchDataset
from ..model import load
from ..infer.tikz import TikzDocument
from ..evaluate.imagesim import ImageSim
from ..util import SplitEpochSaveCallback
from .detikzify_grpo import DetikzifyGRPOTrainer

logger = logging.get_logger("grpo_trainer")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1)) # to be clarified
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

def rl_tokenize(
    batch,
    processor,
    **kwargs
):
    image_token = processor.image_token
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    input_ids = processor(
        #text=[""]*len(batch), #batch['text'],
        images=batch['image'],
        max_length=processor.tokenizer.model_max_length,
        pad_to_multiple_of=8,
        add_eos_token=True,
        **kwargs
    )
    input_ids['labels'] = deepcopy(input_ids['input_ids'])
    #input_ids['text'] = [""]*len(images), #deepcopy(batch['text'])
    input_ids['image'] = deepcopy(batch['image'])

    # do not train on image and pad tokens
    for label_ids in input_ids['labels']:
        for idx, label_id in enumerate(label_ids):
            if label_id in {image_token_id, processor.tokenizer.pad_token_id}:
                label_ids[idx] = IGNORE_INDEX

    return input_ids

    
class ImageRLDataset(ImageSketchDataset):
    """
    Dataset for RL fine-tuning with DeTikZify.
    Provides samples with 'prompt' (image + optional text) and 'completion' (TikZ code).
    """
    def __init__(self, dataset, processor):
        super().__init__(dataset, processor)

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, batch):
        for idx, sketches in enumerate(batch['image']):
            if isinstance(batch["image"][idx], Image.Image):  # Check if it's a valid image
            	batch["image"][idx] = batch["image"][idx].convert("RGB")  # Ensure RGB format

        return rl_tokenize(
            batch=batch,
            processor=self.processor,
            return_tensors="pt",
            truncation=False,
            padding=True
        )

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
            
        prompt = ""
        prompt_image = image
        
        completion = sample.get("text", "")

        return {
            "prompt": prompt,
            "prompt_image": prompt_image,
            "completion": completion,
        }

class RewardFunction:
    def __init__(self, model, processor):
        self.reward_model = ImageSim.from_detikzify(model, processor)
        self.reward_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
    num_epochs: int = 1,
    learning_rate: float = 1e-6,
    gradient_checkpointing: bool = False,
    freeze_vision_enc: bool = False,
    num_gen: int = 8,
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    if WORLD_SIZE != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // WORLD_SIZE

    dataset = ImageRLDataset(dataset, processor)
    #logger.info(f"Dataset size before filtering out too long examples: {len(dataset)}")
    eos_token_id, model_max_length = processor.tokenizer.eos_token_id, processor.tokenizer.model_max_length
    #dataset.filter(lambda ex: (ex['input_ids'] == eos_token_id).nonzero(as_tuple=True)[0].numel() > 0 and (ex['input_ids'] == eos_token_id).nonzero(as_tuple=True)[0][0].item() < model_max_length)
    #logger.info(f"Dataset size after filtering out too long examples: {len(dataset)}")

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

    # freeze the vision encoder parameters
    if freeze_vision_enc:
        logger.info("The vision encoder was frozen. To unfreeze, please start training with --freeze_vision_encoder=False.")
        for _, param in model.model.vision_model.named_parameters():
            param.requires_grad = False

    reward_fn = RewardFunction(model, processor)

    training_args = GRPOConfig(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        warmup_ratio=0.03,
        weight_decay=0,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        #torch_compile=True,
        torch_compile=False,
        bf16=True,
        tf32=True,
        logging_steps=10,
        lr_scheduler_type="cosine",
        optim="adamw_torch" if deepspeed else "adamw_torch_fused",
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=250,
        report_to="none",
        save_total_limit=1,
        output_dir=output_dir,
        deepspeed=deepspeed,

        max_completion_length=2048,
        num_generations=num_gen,
    )

    trainer = DetikzifyGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
        callbacks=[SplitEpochSaveCallback(step_size=0.25)],
    )

    import numpy.core.multiarray
    import torch.serialization
    torch.serialization.add_safe_globals({numpy.core.multiarray._reconstruct})

    trainer.train(resume_from_checkpoint=last_checkpoint)

    if trainer.is_deepspeed_enabled:
        # https://huggingface.co/docs/accelerate/v0.11.0/en/deepspeed#saving-and-loading
        from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
        last_checkpoint = get_last_checkpoint(output_dir)
        load_state_dict_from_zero_checkpoint(trainer.model.float(), last_checkpoint)

    trainer.save_model(output_dir)

    return model, processor