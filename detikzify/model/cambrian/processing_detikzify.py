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
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from transformers.utils import logging
from open_clip import image_transform, create_model
import os

from .encoder.base_encoder import ProcessorWrapper

logger = logging.get_logger(__name__)


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

    def __init__(self, image_processor, tokenizer=None, image_seq_len=300, image_token="<image>",
                 mm_vision_tower_aux_list=None, query_num_list=None):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if image_token not in tokenizer.vocab:
            raise ValueError(f"{image_token} needs to be added to the `tokenizer` vocabulary.")

        super().__init__(image_processor, tokenizer)

        self.image_token = image_token
        self.image_seq_len = image_seq_len or 300

        self.vision_tower_list = mm_vision_tower_aux_list or ["siglip"]
        self.query_num_list = query_num_list or [4] * len(self.vision_tower_list)

        # Store processors and encoders per tower
        self.vision_tower_processors = []
        self.vision_tower_encoders = []

        for tower in self.vision_tower_list:
            if "convnext" in tower.lower():
                transform = image_transform(image_size=1024, is_train=False)
                processor = ProcessorWrapper(transform)
                encoder = create_model(
                    model_name="convnext_large_d_320",
                    pretrained="laion2b_s29b_b131k_ft_soup",
                    precision="fp32"
                ).visual
    
            else:
                processor = AutoImageProcessor.from_pretrained(tower)
                encoder = AutoModel.from_pretrained(tower)
    
            self.vision_tower_processors.append(processor)
            self.vision_tower_encoders.append(encoder)

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: Union[List, torch.Tensor] = None,
        image_seq_len: Optional[int] = None,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        return_image_latents: bool = False,
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
            return BatchFeature(data={**text_inputs, "vision_latents": vision_latents})

        image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        return BatchFeature(data={**text_inputs, **image_inputs})

    def extract_vision_latents(self, images):
        images = make_list_of_images(images)
        vision_latents = []

        for processor, encoder in zip(self.vision_tower_processors, self.vision_tower_encoders):
            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(encoder.device)
            with torch.no_grad():
                features = encoder(pixel_values).last_hidden_state
            vision_latents.append(features)
            
        return vision_latents

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "image_token": self.image_token,
            "image_seq_len": self.image_seq_len,
            "query_num_list": self.query_num_list,
            "mm_vision_tower_aux_list": self.vision_tower_list,
        })
        return base_dict
    
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))
