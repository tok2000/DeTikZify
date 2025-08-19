# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from
# https://github.com/huggingface/transformers/commit/e1b150862e66e16acf951edfa13206ffcd1032be

from typing import List, Optional, Union, Unpack

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_list_of_images
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel, Dinov2Model, CLIPVisionModel, CLIPImageProcessor
from transformers.utils import logging
from open_clip import image_transform, create_model, create_model_and_transforms
import os
import json

from .encoder.base_encoder import ProcessorWrapper

logger = logging.get_logger(__name__)

def load_vision_processors_from_config(config): # added after sanity check
        processors = []
        for tower_type, image_size, tower_name in zip(config["types"], config["image_sizes"], config["towers"]):
            if tower_type in ["clip_convnext"]:
                transform = image_transform(image_size=image_size, is_train=False)
                processors.append(ProcessorWrapper(transform))
            elif tower_type in ["siglip", "dino"]:
                processors.append(AutoImageProcessor.from_pretrained(tower_name))
            elif tower_type in ["clip"]:
                processors.append(CLIPImageProcessor.from_pretrained(tower_name))
            else:
                raise ValueError(f"Unsupported tower_type: {tower_type}")
        return processors
    
def load_vision_encoders_from_config(config): # added after sanity check
    encoders = []
    for tower_type, tower_name in zip(config["types"], config["towers"]):
        if tower_type == "clip":
            # ViT-L/14-336
            encoder = CLIPVisionModel.from_pretrained(tower_name)
        elif tower_type == "clip_convnext":
            if "xxl" in tower_name.lower():
                encoder = create_model("convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup", precision="fp32").visual
            else:
                encoder = create_model("convnext_large_d_320", pretrained="laion2b_s29b_b131k_ft_soup", precision="fp32").visual
        elif tower_type == "siglip":
            encoder = AutoModel.from_pretrained(tower_name).vision_model
        elif tower_type == "dino":
            encoder = AutoModel.from_pretrained(tower_name)
        else:
            raise ValueError(f"Unsupported tower_type: {tower_type}")
        encoders.append(encoder)
    return encoders


class DetikzifyProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": False,
        },
    }


class DetikzifyCambrianProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer=None, image_seq_len=300, image_token="<image>", mm_vision_tower_aux_list=None):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if image_token not in tokenizer.vocab:
            raise ValueError(f"{image_token} needs to be added to the `tokenizer` vocabulary.")

        super().__init__(image_processor, tokenizer)

        self.image_token = image_token
        self.image_seq_len = image_seq_len

        self.vision_tower_list = mm_vision_tower_aux_list or ["google/siglip-so400m-patch14-384"]
        
        # Store processors and encoders per tower
        self.vision_tower_processors = []
        self.vision_tower_encoders = []
        self.mm_vision_tower_types = []
        self.mm_vision_tower_image_sizes = []
        self.mm_vision_tower_names_or_paths = []

        for tower in self.vision_tower_list:
            if "clip-convnext-l" in tower.lower():
                # CLIP-convnext-L
                transform = image_transform(image_size=1024, is_train=False)
                processor = ProcessorWrapper(transform)
                encoder = create_model(
                    model_name="convnext_large_d_320",
                    pretrained="laion2b_s29b_b131k_ft_soup",
                    precision="fp32"
                ).visual
                tower_type = "clip_convnext"
                image_size = 1024
        
            elif "clip-convnext-xxl" in tower.lower():
                # CLIP-convnext-XXL
                transform = image_transform(image_size=1024, is_train=False)
                processor = ProcessorWrapper(transform)
                encoder = create_model(
                    model_name="convnext_xxlarge",
                    pretrained="laion2b_s34b_b82k_augreg_soup",
                    precision="fp32"
                ).visual
                tower_type = "clip_convnext"
                image_size = 1024
        
            elif "clip-vit-large" in tower.lower():
                # ViT-L/14-336
                processor = CLIPImageProcessor.from_pretrained(tower)
                encoder = CLIPVisionModel.from_pretrained(tower)
                tower_type = "clip"
                image_size = 336
    
            elif "siglip" in tower.lower():
                processor = AutoImageProcessor.from_pretrained(tower)
                encoder = AutoModel.from_pretrained(tower).vision_model
                tower_type = "siglip"
                image_size = 384
                
            elif "dino" in tower.lower():
                processor = AutoImageProcessor.from_pretrained(tower)
                encoder = AutoModel.from_pretrained(tower)
                tower_type = "dino"
                image_size = 224
                
            else:
                raise ValueError(f"Unsupported tower_type: {tower}")
    
            self.vision_tower_processors.append(processor)
            self.vision_tower_encoders.append(encoder)
            self.mm_vision_tower_types.append(tower_type)
            self.mm_vision_tower_image_sizes.append(image_size)
            self.mm_vision_tower_names_or_paths.append(tower)

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: Union[List, torch.Tensor] = None,
        image_seq_len: Optional[int] = None,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        return_image_latents: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> BatchFeature:
        if images is None:
            raise ValueError("`images` are expected as arguments to a `DetikzifyProcessor` instance.")
        images = make_list_of_images(images)

        if text is None:
            text = len(images) * [""]
        elif isinstance(text, str):
            text = [text]
        if len(images) != len(text):
            raise ValueError(
                f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
            )

        prompt_strings = []
        for prompt in text:
            if self.image_token in prompt:
                raise ValueError("Image tokens are added by the processor automatically.")
            if add_bos_token:
                prompt = self.tokenizer.bos_token + prompt
            if add_eos_token:
                prompt += self.tokenizer.eos_token
            seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
            prompt_strings.append((self.image_token * seq_len) + prompt)

        text_inputs = self.tokenizer(prompt_strings, return_tensors=return_tensors, padding=True, truncation=True)

        if return_image_latents:
            vision_latents = self.extract_vision_latents(images)
            return BatchFeature(data={**text_inputs, "image_hidden_states": vision_latents})

        image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        return BatchFeature(data={**text_inputs, **image_inputs})

    def extract_vision_latents(self, images): # method completely revised after sanity check
        #images = make_list_of_images(images) # not needed after sanity check
        vision_latents = []

        for processor, encoder in zip(self.vision_tower_processors, self.vision_tower_encoders):
            if isinstance(processor, ProcessorWrapper):
                pixel_values = processor(images=images, return_tensors="pt")['pixel_values'].to(next(encoder.parameters()).device)
                with torch.no_grad():
                    output = encoder(pixel_values)
            else:
                processed = processor(images=images, return_tensors="pt")
                for k in processed:
                    processed[k] = processed[k].to(next(encoder.parameters()).device)
                with torch.no_grad():
                    output = encoder(**processed)
            
            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, "last_hidden_state"):
                features = output.last_hidden_state
            elif isinstance(output, dict) and "last_hidden_state" in output:
                features = output["last_hidden_state"]
            else:
                raise ValueError("Unsupported encoder output format")
                
            vision_latents.append(features)
            
        return vision_latents

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "image_token": self.image_token,
            "image_seq_len": self.image_seq_len,
            "mm_vision_tower_aux_list": self.vision_tower_list,
        })
        return base_dict

    def save_pretrained(self, save_directory, **kwargs): # added after sanity check
        super().save_pretrained(save_directory, **kwargs)
        tower_config = {
            "types": self.mm_vision_tower_types,
            "image_sizes": self.mm_vision_tower_image_sizes,
            "towers": self.mm_vision_tower_names_or_paths,
        }
        tower_config_path = os.path.join(save_directory, "vision_tower_config.json")
        with open(tower_config_path, "w") as f:
            json.dump(tower_config, f)
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs): # added after sanity check
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        tower_config_path = os.path.join(pretrained_model_name_or_path, "vision_tower_config.json")
        if os.path.exists(tower_config_path):
            with open(tower_config_path, "r") as f:
                tower_config = json.load(f)
            processor.vision_tower_processors = load_vision_processors_from_config(tower_config)
            processor.vision_tower_encoders = load_vision_encoders_from_config(tower_config)
            processor.mm_vision_tower_types = tower_config["types"]
            processor.mm_vision_tower_image_sizes = tower_config["image_sizes"]
            processor.mm_vision_tower_names_or_paths = tower_config["towers"]
        return processor

