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
from transformers.image_utils import make_list_of_images
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers import AutoImageProcessor
from transformers.utils import logging
import os
import json
from types import SimpleNamespace

from .encoder.base_encoder import ProcessorWrapper
from .encoder.clip_encoder import ClipVisionTower
from .encoder.siglip_encoder import SiglipVisionTower
from .encoder.dino_encoder import DinoVisionTower
from .encoder.clip_convnext_encoder import CLIPConvNextTower

logger = logging.get_logger(__name__)

# load processors from config file
def load_vision_processors_from_config(config):
    processors = []
    for tower_type, image_size, tower_name in zip(config["types"], config["image_sizes"], config["towers"]):
        if tower_type == "clip":
            encoder = ClipVisionTower(tower_name, args={})
            processor = encoder.image_processor
        elif tower_type == "clip_convnext":
            encoder = CLIPConvNextTower(tower_name, args=SimpleNamespace(mm_vision_select_layer=12, mm_vision_select_feature="patch", unfreeze_mm_vision_tower=False), delay_load=True)
            encoder.load_model()
            processor = encoder.image_processor
        elif tower_type == "siglip":
            encoder = SiglipVisionTower(tower_name, args={})
            processor = encoder.image_processor
        elif tower_type == "dino":
            encoder = DinoVisionTower(tower_name, args={})
            processor = encoder.image_processor
        else:
            raise ValueError(f"Unsupported tower_type: {tower_type}")
        processors.append(processor) # add processor to list
    return processors # return list of processors

# load encoders from config file
def load_vision_encoders_from_config(config):
    encoders = []
    for tower_type, tower_name in zip(config["types"], config["towers"]):
        if tower_type == "clip":
            encoder = ClipVisionTower(tower_name, args={})
        elif tower_type == "clip_convnext":
            encoder = CLIPConvNextTower(tower_name, args=SimpleNamespace(mm_vision_select_layer=12, mm_vision_select_feature="patch", unfreeze_mm_vision_tower=False), delay_load=True)
            encoder.load_model()
        elif tower_type == "siglip":
            encoder = SiglipVisionTower(tower_name, args={})
        elif tower_type == "dino":
            encoder = DinoVisionTower(tower_name, args={})
        else:
            raise ValueError(f"Unsupported tower_type: {tower_type}")
        encoders.append(encoder) # add encoder to list
    return encoders # return list of encoders


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

    def __init__(self, image_processor, tokenizer=None, image_seq_len=300, image_token="<|reserved_special_token_2|>", mm_vision_tower_aux_list=None, vision_encoders=None, original_tower_names=None):
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if image_token not in tokenizer.vocab:
            raise ValueError(f"{image_token} needs to be added to the `tokenizer` vocabulary.")

        if image_processor is None: # default to siglip processor
            image_processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        super().__init__(image_processor, tokenizer)

        self.image_token = image_token
        self.image_seq_len = image_seq_len

        # Use mm_vision_tower_aux_list for backward compatibility
        self.vision_tower_list = mm_vision_tower_aux_list or ["google/siglip-so400m-patch14-384"]
        
        # Store processors and encoders per tower
        self.vision_tower_processors = []
        self.vision_tower_encoders = []
        self.mm_vision_tower_types = []
        self.mm_vision_tower_image_sizes = []
        self.mm_vision_tower_names_or_paths = []

        # Use pre-loaded encoders if provided, otherwise load them
        if vision_encoders is not None:
            self.vision_tower_encoders = vision_encoders
            
            # Use original tower names if provided, otherwise extract from encoders
            if original_tower_names is not None and len(original_tower_names) == len(vision_encoders):
                tower_names = original_tower_names
            else:
                tower_names = []
                for encoder in vision_encoders: # try to extract tower name from encoder
                    if hasattr(encoder, 'model_name_or_path'):
                        tower_names.append(encoder.model_name_or_path)
                    else:
                        tower_names.append(str(type(encoder)))
            
            # load processors based on tower names
            for encoder, tower_name in zip(vision_encoders, tower_names):
                self.vision_tower_processors.append(encoder.image_processor)
                
                if "clip-vit-large" in tower_name.lower():
                    tower_type = "clip"
                elif "siglip" in tower_name.lower():
                    tower_type = "siglip"
                elif "dino" in tower_name.lower():
                    tower_type = "dino"
                elif "clip-convnext" in tower_name.lower():
                    tower_type = "clip_convnext"
                else:
                    tower_type = "unknown"
                
                self.mm_vision_tower_types.append(tower_type)
                self.mm_vision_tower_image_sizes.append(encoder.image_size)
                self.mm_vision_tower_names_or_paths.append(tower_name)
        else: # load encoders and processors based on tower list
            for tower in self.vision_tower_list:
                if "clip-vit-large" in tower.lower():
                    encoder = ClipVisionTower(tower, args={})
                    processor = encoder.image_processor
                    tower_type = "clip"
                    image_size = encoder.image_size
                elif "siglip" in tower.lower():
                    encoder = SiglipVisionTower(tower, args={})
                    processor = encoder.image_processor
                    tower_type = "siglip"
                    image_size = encoder.image_size
                elif "dino" in tower.lower():
                    encoder = DinoVisionTower(tower, args={})
                    processor = encoder.image_processor
                    tower_type = "dino"
                    image_size = encoder.image_size
                elif "clip-convnext" in tower.lower():
                    encoder = CLIPConvNextTower(tower, args=SimpleNamespace(mm_vision_select_layer=12, mm_vision_select_feature="patch", unfreeze_mm_vision_tower=False), delay_load=True)
                    encoder.load_model()
                    processor = encoder.image_processor
                    tower_type = "clip_convnext"
                    image_size = encoder.image_size
                else:
                    raise ValueError(f"Unsupported tower_type: {tower}")
            
                # save processor and encoder and metadata
                self.vision_tower_processors.append(processor)
                self.vision_tower_encoders.append(encoder)
                self.mm_vision_tower_types.append(tower_type)
                self.mm_vision_tower_image_sizes.append(image_size)
                self.mm_vision_tower_names_or_paths.append(tower)

        # Validate that all encoders produce the same number of tokens
        if len(self.vision_tower_encoders) > 1:
            with torch.no_grad():
                dummy_image = torch.randn(1, 3, 384, 384)
                token_counts = []
                for encoder in self.vision_tower_encoders:
                    features = encoder(dummy_image)
                    token_counts.append(features.shape[1])  # Get token count
                
                if not all(tc == token_counts[0] for tc in token_counts):
                    raise ValueError(f"All vision encoders must produce the same number of tokens. Got: {token_counts}")

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: Union[List, torch.Tensor] = None,
        image_seq_len: Optional[int] = None,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        return_image_latents: bool = True,
        return_pixel_values: bool = False,
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

        # return image latents from multi-vision towers if requested
        if return_image_latents:
            vision_latents = self.extract_vision_latents(images)
            return BatchFeature(data={**text_inputs, "image_hidden_states": vision_latents}) # include list of vision latents


        if return_pixel_values:
            multi_pixel_values = []
            for processor, encoder in zip(self.vision_tower_processors, self.vision_tower_encoders):
                if hasattr(processor, "preprocess") or hasattr(processor, "image_mean"):
                    processed = processor(images=images, return_tensors="pt")
                    pixel_values = processed['pixel_values']
                    # Resize pixel values to match encoder's expected input size
                    if hasattr(encoder, '_image_size') and pixel_values.shape[-1] != encoder._image_size:
                        pixel_values = torch.nn.functional.interpolate(
                            pixel_values, 
                            size=(encoder._image_size, encoder._image_size), 
                            mode="bilinear", 
                            align_corners=False
                        )
                elif hasattr(processor, "image_processor"):  # Handle processors with image_processor attribute
                    processed = processor.image_processor(images=images, return_tensors="pt")
                    pixel_values = processed['pixel_values']
                    # Resize pixel values to match encoder's expected input size
                    if hasattr(encoder, '_image_size') and pixel_values.shape[-1] != encoder._image_size:
                        pixel_values = torch.nn.functional.interpolate(
                            pixel_values, 
                            size=(encoder._image_size, encoder._image_size), 
                            mode="bilinear", 
                            align_corners=False
                        )
                else:
                    # Assume torchvision transform or ProcessorWrapper
                    # Apply to each image and stack
                    pixel_values = torch.stack([processor(image) for image in images])
                    # Resize pixel values to match encoder's expected input size
                    if hasattr(encoder, '_image_size') and pixel_values.shape[-1] != encoder._image_size:
                        pixel_values = torch.nn.functional.interpolate(
                            pixel_values, 
                            size=(encoder._image_size, encoder._image_size), 
                            mode="bilinear", 
                            align_corners=False
                        )
                multi_pixel_values.append(pixel_values)
            return BatchFeature(
                data={
                    **text_inputs,
                    **{f"pixel_values_tower_{i}": v for i, v in enumerate(multi_pixel_values)}
                }
            )
        
        # use the first vision tower processor for fallback
        if self.vision_tower_processors:
            image_inputs = self.vision_tower_processors[0](images=images, return_tensors=return_tensors)
        else: # fallback to default image processor
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        return BatchFeature(data={**text_inputs, **image_inputs})

    # extract vision latents from all vision towers
    def extract_vision_latents(self, images):
        vision_latents = []
        for processor, encoder in zip(self.vision_tower_processors, self.vision_tower_encoders):
            # extract pixel values using the processor
            if hasattr(processor, "preprocess") or hasattr(processor, "image_mean"): # Handle HuggingFace processors
                processed = processor(images=images, return_tensors="pt")
                pixel_values = processed['pixel_values']
            elif hasattr(processor, "image_processor"): # Handle processors with image_processor attribute
                processed = processor.image_processor(images=images, return_tensors="pt")
                pixel_values = processed['pixel_values']
            else:
                # Assume torchvision transform or ProcessorWrapper
                # Apply to each image and stack
                pixel_values = torch.stack([processor(image) for image in images])
            
            # extract features using the encoder
            with torch.no_grad():
                features = encoder(pixel_values)

            vision_latents.append(features)
        return vision_latents # return list of vision latents

    # serialize to dictionary
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "image_token": self.image_token,
            "image_seq_len": self.image_seq_len,
            "mm_vision_tower_aux_list": self.vision_tower_list,
        })
        return base_dict

    # save processor and vision tower config
    def save_pretrained(self, save_directory, **kwargs):
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

    # load processor and vision tower config from pretrained
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
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
        else:
            print(f"[DEBUG] No vision tower config found at {tower_config_path}")
        return processor

