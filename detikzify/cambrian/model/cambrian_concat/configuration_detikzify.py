# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
from typing import Union

from transformers import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class DetikzifyCambrianVisionConfig(PretrainedConfig): # store the configuration of the vision model
    model_type = "detikzify"

    def __init__(
        self,
        vision_towers=None,  # List of vision tower names
        total_hidden_size=None,  # Calculated automatically from towers
        image_size=384,  # Default image size for processing
        concat_factor=3,  # How many image tokens to concatenate
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Handle vision towers input
        if vision_towers is None:
            vision_towers = ["google/siglip-so400m-patch14-384"]
        elif isinstance(vision_towers, str):
            vision_towers = [vision_towers]
        
        if not isinstance(vision_towers, list):
            raise ValueError("vision_towers must be a list of vision tower names")
        
        self.vision_towers = vision_towers
        self.image_size = image_size
        self.concat_factor = concat_factor
        
        # Calculate total hidden size from vision towers
        if total_hidden_size is None:
            total_hidden_size = self.calculate_total_hidden_size()
        
        self.total_hidden_size = total_hidden_size
        
        print(f"Created vision config with {len(self.vision_towers)} towers, total_hidden_size: {self.total_hidden_size}")

    def calculate_total_hidden_size(self):
        total_hidden_size = 0
        for vision_tower_name in self.vision_towers:
            if "clip-vit-large" in vision_tower_name.lower():
                total_hidden_size += 1024  # CLIP ViT-Large
            elif "siglip" in vision_tower_name.lower():
                total_hidden_size += 1152  # SigLIP ViT-SO400M
            elif "dino" in vision_tower_name.lower():
                total_hidden_size += 768   # DINO v2 Base
            elif "clip-convnext" in vision_tower_name.lower():
                if "xxl" in vision_tower_name.lower():
                    total_hidden_size += 3072  # CLIP ConvNext-XXL
                else:
                    total_hidden_size += 1536  # CLIP ConvNext-Large
            else:
                # Default fallback
                total_hidden_size += 1024
        return total_hidden_size

    def validate_vision_towers(self):
        """Validate that vision tower names are supported"""
        supported_prefixes = [
            "clip-vit-large", "siglip", "dino", "clip-convnext"
        ]
        
        for tower_name in self.vision_towers:
            if not any(prefix in tower_name.lower() for prefix in supported_prefixes):
                logger.warning(f"Vision tower '{tower_name}' may not be supported. Supported prefixes: {supported_prefixes}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from DetikzifyConfig
        if config_dict.get("model_type") == "detikzify":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class DetikzifyCambrianConfig(PretrainedConfig): # store the overall configuration of the model, including the vision and text configurations
    model_type = "detikzify"
    is_composition = True # indicates that the model is a multi-modal model

    def __init__(
        self,
        use_cache=True, # enable key value caching for faster text generation
        image_token_id=128005, # token ID for the image token
        tie_word_embeddings=False, # if True, input and output word embeddings share the same weights
        vision_config=None, # store the vision configuration
        text_config=None, # store the text configuration
        concat_factor=3, # determines how many image tokens are concatenated to the text tokens
        pad_token_id=128004, # token ID for the padding token

        mm_vision_tower_aux_list=None, # auxiliary vision tower list
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        # Handle vision configuration
        if vision_config is None:
            # Use mm_vision_tower_aux_list for backward compatibility
            vision_towers = mm_vision_tower_aux_list
            if vision_towers is None:
                vision_towers = ["google/siglip-so400m-patch14-384"]
            elif isinstance(vision_towers, str):
                vision_towers = [vision_towers]
            
            self.vision_config = DetikzifyCambrianVisionConfig(
                vision_towers=vision_towers,
                concat_factor=concat_factor
            )
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = DetikzifyCambrianVisionConfig(**vision_config)
        elif isinstance(vision_config, DetikzifyCambrianVisionConfig):
            self.vision_config = vision_config
        else:
            raise ValueError("vision_config must be None, dict, or DetikzifyCambrianVisionConfig")

        # Validate vision towers
        self.vision_config.validate_vision_towers()

        # Handle text configuration
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, using default text config")
            text_config = CONFIG_MAPPING["llama"](
                rms_norm_eps=1e-5,
                pad_token_id=128004,
                tie_word_embeddings=False,
                hidden_size=2048,
                intermediate_size=8192,
                vocab_size=128256,
                num_hidden_layers=16,
                num_attention_heads=32,
                num_key_value_heads=8,
                use_scaled_rope=True,
                rope_theta=500000.0,
                bos_token_id=128000,
                eos_token_id=128001
            )

        self.text_config = text_config
        self.concat_factor = self.vision_config.concat_factor  # Use vision config's concat_factor
        self.seed = kwargs.pop("seed", 42)

        # For backward compatibility, keep mm_vision_tower_aux_list
        self.mm_vision_tower_aux_list = self.vision_config.vision_towers

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

    @property
    def total_vision_hidden_size(self):
        """Get the total hidden size from all vision encoders"""
        return self.vision_config.total_hidden_size

    @property
    def vision_towers(self):
        """Get the list of vision tower names"""
        return self.vision_config.vision_towers

    def _get_non_default_generation_parameters(self):
        # Avoid triggering unwanted default config reconstruction
        return {}
