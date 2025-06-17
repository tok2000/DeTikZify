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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
from transformers import (
    AutoModel,
    Cache,
    DynamicCache,
    GenerationMixin,
    PreTrainedModel,
    SiglipVisionModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from .configuration_detikzify import DetikzifyConfig


logger = logging.get_logger(__name__)


@dataclass
class DetikzifyBaseModelOutputWithPast(ModelOutput): # Base class for model outputs
    last_hidden_state: torch.FloatTensor = None # final hidden states of the model
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None # cached key-value pairs for decoding
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None # hidden states at each layer of the model
    attentions: Optional[Tuple[torch.FloatTensor]] = None # attention scores of each layer of the model
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None # hidden states of the image encoder


@dataclass
class DetikzifyCausalLMOutputWithPast(ModelOutput): # used for causal language modeling outputs
    loss: Optional[torch.FloatTensor] = None # cross-entropy loss for text generation
    logits: torch.FloatTensor = None # logits of the model (output of the decoder)
    past_key_values: Optional[List[torch.FloatTensor]] = None # cached key-value pairs for decoding
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None # hidden states at each layer of the model
    attentions: Optional[Tuple[torch.FloatTensor]] = None # attention scores of each layer of the model
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None # hidden states of the image encoder


class DetikzifySimpleMLP(nn.Module): # simple MLP for modality projection from vision to text
    def __init__(self, config):
        super().__init__()
        input_size = config.vision_config.hidden_size * config.concat_factor # concat_factor determines number of patches to concatenate
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class DetikzifyConnector(nn.Module): # connector module to project image hidden states to text hidden states
    def __init__(self, config):
        super().__init__()
        self.concat_factor = config.concat_factor # number of patches to concatenate
        self.modality_projection = DetikzifySimpleMLP(config) # simple MLP for modality projection

    def concatenate(self, x, concat_factor=3): # reshape image hidden states to concatenate patches
        bsz, seq, embed_dim = x.size()
        return x.reshape(bsz, seq // concat_factor, embed_dim * concat_factor)

    def forward(self, image_hidden_states): # forward pass for the connector module
        image_hidden_states = self.concatenate(image_hidden_states, self.concat_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class DetikzifyPreTrainedModel(PreTrainedModel): # base class for all models
    config_class = DetikzifyConfig # uses DetikzifyConfig for configuration
    base_model_prefix = "model"
    supports_gradient_checkpointing = True # enable gradient checkpointing to save memory during training
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values" # specify keys which should not be moved across devices
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module): # initialize weights of the model using normal distribution
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range") # use initializer_range from config if available
            else self.config.text_config.initializer_range # otherwise use initializer_range from text_config
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std) # initialize class embedding using normal distribution

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std) # initialize weights of linear and convolutional layers using normal distribution
            if module.bias is not None:
                module.bias.data.zero_() # initialize bias to zero
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std) # initialize weights of embedding layers using normal distribution
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_() # initialize padding index to zero


class DetikzifyModel(DetikzifyPreTrainedModel): # main model class for Detikzify
    def __init__(self, config: DetikzifyConfig):
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = SiglipVisionModel._from_config(config.vision_config) # initialize vision model
        self.connector = DetikzifyConnector(config) # initialize connector module
        self.text_model = AutoModel.from_config(config.text_config) # initialize text model

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.concat_factor) # calculate image sequence length
        )
        self.image_token_id = self.config.image_token_id

        self._use_flash_attention_2 = config.text_config._attn_implementation == "flash_attention_2" # use flash_attention_2 if specified in config

        self.post_init()

    def enable_input_require_grads(self): # enable input gradients for text and vision models
        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads) # enable input gradients for text model
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook( # enable input gradients for vision model
            make_inputs_require_grads
        )

    def disable_input_require_grads(self): # disable input gradients for text and vision models
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        if image_hidden_states is None:
            raise ValueError("image_hidden_states cannot be None")
    
        max_img_tokens = 300
    
        # Ensure `image_hidden_states` has the correct number of tokens
        if image_hidden_states.shape[1] < max_img_tokens:
            pad_tokens = torch.zeros(
                (image_hidden_states.shape[0], max_img_tokens - image_hidden_states.shape[1], image_hidden_states.shape[2]),
                device=image_hidden_states.device
            )
            image_hidden_states = torch.cat([image_hidden_states, pad_tokens], dim=1)
        elif image_hidden_states.shape[1] > max_img_tokens:
            image_hidden_states = image_hidden_states[:, :max_img_tokens, :]
    
        # Clone inputs_embeds to avoid in-place errors
        new_inputs_embeds = inputs_embeds.clone()
        reshaped_image_hidden_states = image_hidden_states.view(-1, image_hidden_states.shape[2])
        reshaped_image_hidden_states = reshaped_image_hidden_states.to(inputs_embeds.dtype)
    
        # Get mask of special image tokens
        special_image_token_mask = (input_ids == self.image_token_id).nonzero(as_tuple=True)
    
        num_image_tokens = min(len(special_image_token_mask[0]), max_img_tokens)  # Prevent out-of-bounds access
    
        # **CUDA SAFE FIX: Limit mask size before indexing**
        if num_image_tokens > reshaped_image_hidden_states.shape[0]:
            num_image_tokens = reshaped_image_hidden_states.shape[0]  # Prevent indexing error
    
        if num_image_tokens > 0:
            try:
                new_inputs_embeds[special_image_token_mask[0][:num_image_tokens], special_image_token_mask[1][:num_image_tokens]] = reshaped_image_hidden_states[:num_image_tokens]
            except IndexError as e:
                print(f"ERROR: Indexing mismatch! {e}")
                print(f"special_image_token_mask.shape = {special_image_token_mask[0].shape}")
                print(f"reshaped_image_hidden_states.shape = {reshaped_image_hidden_states.shape}")
                raise
    
        return new_inputs_embeds





    def forward( # forward pass for the Detikzify model
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DetikzifyBaseModelOutputWithPast]: # return output of the model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape # get batch size and sequence length
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape # get batch size, sequence length, and embedding size
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache() # initialize cache for past key values
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None: # if pixel values are provided
            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model( # forward pass for the vision model
                pixel_values=pixel_values.to(dtype=self.dtype),  # fp16 compatibility for mixed precision training
            ).last_hidden_state
            # Modality projection
            image_hidden_states = self.connector(image_hidden_states) # project image hidden states to text hidden states using connector module

        elif image_hidden_states is not None: # if image hidden states are provided
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device) # no need to project image hidden states to text hidden states

        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger( # merge text and image inputs using image hidden states and input embeddings
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model( # forward pass for the text model
            inputs_embeds=inputs_embeds, # contains text and image inputs
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict: # return output of the model without past key values
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return DetikzifyBaseModelOutputWithPast( # return output of the model with past key values
            last_hidden_state=outputs.last_hidden_state, # final hidden states of the model
            past_key_values=outputs.past_key_values, # cached key-value pairs for decoding
            hidden_states=outputs.hidden_states, # hidden states at each layer of the model
            attentions=outputs.attentions, # attention scores of each layer of the model
            image_hidden_states=image_hidden_states, # hidden states of the image encoder
        )


class DetikzifyForConditionalGeneration(DetikzifyPreTrainedModel, GenerationMixin): # model class for conditional generation
    _tied_weights_keys = ["lm_head.weight"] # specify keys for tied weights

    def __init__(self, config):
        super().__init__(config)
        self.model = DetikzifyModel(config) # initialize Detikzify model
        self.image_token_id = self.config.image_token_id

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False) # linear layer for language modeling head
        self.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    def enable_input_require_grads(self): # enable input gradients for text and vision models
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads) # enable input gradients for text model
        self._vision_require_grads_hook = self.model.vision_model.get_input_embeddings().register_forward_hook( # enable input gradients for vision model
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.model.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()

        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings.weight = input_embeddings.weight

    def forward( # forward pass for the conditional generation model
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # cached key-value pairs for decoding
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None, # pixel values for image inputs
        image_hidden_states: Optional[torch.FloatTensor] = None, # hidden states of the image encoder
        labels: Optional[torch.LongTensor] = None, # labels for text generation
        use_cache: Optional[bool] = None, # use cache for decoding
        output_attentions: Optional[bool] = None, # output attention scores
        output_hidden_states: Optional[bool] = None, # output hidden states
        return_dict: Optional[bool] = None, # return output as a dictionary
    ) -> Union[Tuple, DetikzifyCausalLMOutputWithPast]: # return output of the model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model( # forward pass for the Detikzify generator model
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids, # position ids for text inputs
            past_key_values=past_key_values, # cached key-value pairs for decoding
            inputs_embeds=inputs_embeds, # input embeddings for text inputs
            pixel_values=pixel_values, # pixel values for image inputs
            image_hidden_states=image_hidden_states, # hidden states of the image encoder
            use_cache=use_cache, # use cache for decoding
            output_attentions=output_attentions, # output attention scores
            output_hidden_states=output_hidden_states, # output hidden states
            return_dict=return_dict, # return output as a dictionary
        )

        hidden_states = outputs[0] # get hidden states from the transformer model
        logits = self.lm_head(hidden_states) # get logits from the language modeling head
        logits = logits.float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device) # move labels to the same device as logits
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device) # shift attention mask
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous() # remove last token from logits
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous() # remove first token from labels
            else:
                shift_logits = logits[..., :-1, :].contiguous() # remove last token from logits
                shift_labels = labels[..., 1:].contiguous() # remove first token from labels
            # Flatten the tokens
            loss_fct = CrossEntropyLoss() # use cross-entropy loss for text generation
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # calculate loss using logits and labels

        if not return_dict:
            output = (logits,) + outputs[1:] # return logits and other outputs from the model
            return (loss,) + output if loss is not None else output

        return DetikzifyCausalLMOutputWithPast( # return output as a dictionary
            loss=loss, # cross-entropy loss for text generation
            logits=logits, # logits of the model (output of the decoder)
            past_key_values=outputs.past_key_values, # cached key-value pairs for decoding
            hidden_states=outputs.hidden_states, # hidden states at each layer of the model
            attentions=outputs.attentions, # attention scores of each layer of the model
            image_hidden_states=outputs.image_hidden_states, # hidden states of the image encoder
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None, # defines which tokens should be attended to
        inputs_embeds=None, # precomputed embeddings
        cache_position=None, # position of the last token processed in the cache
        pixel_values=None,
        image_hidden_states=None,
        num_logits_to_keep=None, # controls the number of logits to keep for each generation step
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        if past_key_values is not None: # if past key values exist, only keep unprocessed tokens
            if inputs_embeds is not None:  # remove processed tokens from input embeddings
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]: # if using input ids, remove processed tokens based on cache position
                input_ids = input_ids[:, cache_position]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1 # ensures that position ids start from 0
            position_ids.masked_fill_(attention_mask == 0, 1) # mask padding tokens get fixed position ids
            if past_key_values: # if past key values exist, only keep unprocessed tokens
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # but IDEFICS requires both ids and embeds to be present
        if inputs_embeds is not None and cache_position[0] == 0: # if input embeddings are provided, only use them in the first generation step
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": input_ids}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}
        if num_logits_to_keep is not None: # specify number of logits to keep for each generation step, useful for beam search
            model_inputs["num_logits_to_keep"] = num_logits_to_keep
        if image_hidden_states is not None:
            pixel_values = None # if image hidden states are provided, pixel values are not needed

        model_inputs.update( # update model inputs with all the necessary information
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_hidden_states": image_hidden_states,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs): # update model kwargs after each generation step
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs, # model outputs from the previous generation step
            model_kwargs=model_kwargs, # dictionary of input arguments
            is_encoder_decoder=is_encoder_decoder, # whether the model is an encoder-decoder model
            **kwargs,
        )
        # Get the precomputed image_hidden_states
        model_kwargs["image_hidden_states"] = outputs.image_hidden_states # ensure that image hidden states are passed to the model and persist across generation steps
        return model_kwargs
