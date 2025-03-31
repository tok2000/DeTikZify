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


class DetikzifyVisionConfig(PretrainedConfig): # store the configuration of the vision model
    model_type = "detikzify"

    def __init__(
        self,
        hidden_size=1152, # embedding dimension
        intermediate_size=4304, # intermediate size of the feed-forward layer in the encoder
        num_hidden_layers=27, # number of hidden layers in the encoder
        num_attention_heads=16, # number of attention heads in the encoder
        num_channels=3, # number of channels in the input image (RGB)
        image_size=420, # size of the input image (image_size x image_size)
        patch_size=14, # size of each patch in the input image
        hidden_act="gelu_pytorch_tanh", # activation function used in the encoder
        layer_norm_eps=1e-6, # epsilon value for layer normalization
        attention_dropout=0.0, # dropout rate for the attention layers
        initializer_range=0.02, # standard deviation for the weight initialization
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig": # load the configuration from a pretrained model
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs) # fetch the configuration dictionary from the pretrained model

        # get the vision config dict if we are loading from DetikzifyConfig
        if config_dict.get("model_type") == "detikzify":
            config_dict = config_dict["vision_config"] # get only the vision configuration

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning( # log a warning if the model type is different from the DetikzifyVisionConfig model type
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
        mm_projector_type="sva", # type of projector used in the multimodal model
        query_num_list=None, # number of queries for each vision tower
        connector_depth=3, # depth of the connector
        vision_hidden_size=1024, # embedding dimension of the vision model
        sva_layers=2, # number of spatial vision aggregator layers

        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        if mm_vision_tower_aux_list is None:
            mm_vision_tower_aux_list = ["siglip"]
        elif isinstance(mm_vision_tower_aux_list, str):
            mm_vision_tower_aux_list = [mm_vision_tower_aux_list]

        self.mm_vision_tower_aux_list = mm_vision_tower_aux_list

        if not isinstance(self.mm_vision_tower_aux_list, list):
            raise ValueError("mm_vision_tower_aux_list must be a list of vision tower names")

        print(f"mm_vision_tower_aux_list: {self.mm_vision_tower_aux_list}")  # Debugging
        print(f"query_num_list: {query_num_list}")  # Debugging

        if query_num_list is None:
            logger.warning("query_num_list is missing in config.json. Using default.")
            query_num_list = [4] * len(self.mm_vision_tower_aux_list)

        self.query_num_list = query_num_list

        # ensures that the length of the query_num_list is the same as the length of the mm_vision_tower_aux_list
        if len(self.query_num_list) != len(self.mm_vision_tower_aux_list):
            raise ValueError(
                f"query_num_list must have the same length as mm_vision_tower_aux_list, "
                f"got {len(self.query_num_list)} and {len(self.mm_vision_tower_aux_list)}"
            )

        # dynamically calculate the vision_hidden_size based on the vision tower names
        if len(self.mm_vision_tower_aux_list) == 1:
            enc = self.mm_vision_tower_aux_list[0]
            self.vision_hidden_size = 1024 if "siglip" in enc else 1152 if "dino" in enc else 1536 if "CLIP-convnext_l" in enc else 3072 if "CLIP-convnext_xxl" in enc else 1152
        else:
            self.vision_hidden_size = sum(
                1024 if "siglip" in enc
                else 1152 if "dino" in enc
                else 1536 if "CLIP-convnext_l" in enc
                else 3072 if "CLIP-convnext_xxl" in enc
                else 1152
                for enc in self.mm_vision_tower_aux_list
            )

        self.mm_projector_type = mm_projector_type
        self.query_num_list = query_num_list
        self.connector_depth = connector_depth
        self.sva_layers = sva_layers

        if vision_config is None: # if no vision configuration is provided, use the default vision configuration
            self.vision_config = DetikzifyVisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict): # if a dictionary is provided, convert it to a DetikzifyVisionConfig object
            self.vision_config = DetikzifyVisionConfig(**vision_config)
        elif isinstance(vision_config, DetikzifyVisionConfig): # if a DetikzifyVisionConfig object is provided, use it as is
            self.vision_config = vision_config

        if isinstance(text_config, dict): # if a dictionary is provided, convert it to a text configuration object
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama" # set the model type to "LLaMA" as default if not provided
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None: # if no text configuration is provided, use the default text configuration (LLaMA)
            logger.info("text_config is None, using default text config")
            text_config = CONFIG_MAPPING["llama"]( # use the default LLaMA configuration
                rms_norm_eps=1e-5, # epsilon value for RMS normalization
                pad_token_id=128004, # token ID for the padding token
                tie_word_embeddings=False,

                hidden_size=2048, # embedding dimension of LLaMA 3.2 1B
                intermediate_size=8192, # intermediate size of the feed-forward layer in LLaMA 3.2 1B
                vocab_size=128256, # vocabulary size of LLaMA 3.2 1B
                num_hidden_layers=16, # number of hidden layers in LLaMA 3.2 1B
                num_attention_heads=32, # number of attention heads in LLaMA 3.2 1B
                num_key_value_heads=8, # number of key-value attention heads in LLaMA 3.2 1B
                use_scaled_rope=True, # use scaled relative positional encoding
                rope_theta=500000.0, # theta value for the relative positional encoding
                bos_token_id=128000, # token ID for the beginning of sentence token
                eos_token_id=128001 # token ID for the end of sentence token
            )

        self.text_config = text_config
        self.concat_factor = concat_factor

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)
