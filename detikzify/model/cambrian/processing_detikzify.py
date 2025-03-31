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
from transformers.utils import logging

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

    def __init__(self, image_processor, tokenizer=None, image_seq_len: int = 300, image_token: str = "<|reserved_special_token_2|>", **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if image_token not in tokenizer.vocab:
            raise ValueError(f"{image_token} needs to be added to the `tokenizer` vocabulary.")

        self.image_token = image_token # store the image token
        self.image_seq_len = image_seq_len if image_seq_len is not None else 300 # store the image sequence length
        self.vision_tower_list = kwargs.get("mm_vision_tower_aux_list", ["siglip"]) # store the vision tower list

        super().__init__(image_processor, tokenizer, **kwargs) # initialize the processor calling the ProcessorMixin class

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None, # accept text or pre-tokenized input
        images: ImageInput = None, # accept images
        image_seq_len: Optional[int] = None, # override the image sequence length
        add_bos_token: bool = None, # add beginning of sentence token
        add_eos_token: bool = None, # add end of sentence token
        return_image_latents: bool = False, # option to return image latents
        **kwargs: Unpack[DetikzifyProcessorKwargs],
    ) -> BatchEncoding:
        output_kwargs = self._merge_kwargs( # merges keyword arguments with default tokenizer and image processor arguments
            DetikzifyProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # Temporary fix for "padding_side" in init_kwargs
        output_kwargs["text_kwargs"].pop("padding_side", None) # remove padding_side from text_kwargs

        if images is None: # if no images are provided, raise an error
            raise ValueError("`images` are expected as arguments to a `DetikzifyProcessor` instance.")
        else:
            images = make_list_of_images(images) # convert images to a list of images to enable batch processing
        if text is None: # if no text is provided, default to an empty string
            text = len(images) * [""]
        elif isinstance(text, str): # if text is a single string, convert it to a list of strings
            text = [text]
        if len(images) != len(text): # ensure equal number of images and text prompts
            raise ValueError(
                f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
            )

        prompt_strings = []
        for prompt in text:
            assert self.image_token not in prompt, "Image tokens are added by the processor!"
            if add_bos_token: # add beginning of sentence token
                prompt = self.tokenizer.bos_token + prompt
            if add_eos_token: # add end of sentence token
                prompt += self.tokenizer.eos_token
            image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
            prompt_strings.append((self.image_token * image_seq_len) + prompt) # append image token to prompt

        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"]) # process images
        text_inputs = self.tokenizer(text=prompt_strings, **output_kwargs["text_kwargs"]) # tokenize text prompts

        if return_image_latents:
            vision_latents = self.extract_vision_latents(images) # extract vision latents
            return BatchFeature(data={**text_inputs, "vision_latents": vision_latents}) # return combined image and text inputs

        return BatchFeature(data={**image_inputs, **text_inputs}) # return combined image and text inputs
    
    def extract_vision_latents(self, images: ImageInput): # extract vision latents
        image_features = []
        images = make_list_of_images(images) # convert images to a list of images to enable batch processing

        if isinstance(self.vision_tower_list, list) and len(self.vision_tower_list) > 1:
            for vision_tower in self.vision_tower_list:
                vision_features = self.image_processor(images=images, return_tensors=True) # process images
                image_features.append(vision_features)
            return torch.cat(image_features, dim=-1) # concatenate vision features along the last dimension
        else:
            return self.image_processor(images=images, return_tensors=False) # single vision tower case

    def batch_decode(self, *args, **kwargs): # decode batch of inputs
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs): # decode single input
        return self.tokenizer.decode(*args, **kwargs)

    @property # ensures the model receives all required inputs
    def model_input_names(self): # returns a list of all input names required by the model
        tokenizer_input_names = self.tokenizer.model_input_names # get input names from tokenizer
        image_processor_input_names = self.image_processor.model_input_names # get input names from image processor
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
