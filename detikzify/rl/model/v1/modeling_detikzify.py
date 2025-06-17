# Adopted from https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava.py. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from pickle import UnpicklingError
from typing import List, Optional, Tuple, Union

from numpy import clip
from safetensors.torch import load_file
from timm import create_model as create_vision_model
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import (
    BatchEncoding,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

from .configuration_detikzify import DetikzifyConfig
from .processing_detikzify import DetikzifyImageProcessor


logger = logging.get_logger("transformers")


class DetikzifyVisionModel(PreTrainedModel):
    _no_split_modules = ["VisionTransformer"]
    def __init__(self, model, **kwargs) -> None:
        super().__init__(PretrainedConfig.from_dict(model.pretrained_cfg), **kwargs)
        # HACK: wrap in list so that vision model does not count as a parameter
        self.model = [model]

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model[0].patch_embed

    def to_input_dtype(self, pixel_values: torch.Tensor):
        target_dtype = self.get_input_embeddings().proj.weight.dtype
        return pixel_values.to(dtype=target_dtype)

    def forward(self, pixel_values: torch.Tensor):
        last_hidden_state = self.model[0].forward_features(self.to_input_dtype(pixel_values))
        pooler_output = self.model[0].forward_head(last_hidden_state)
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output
        )

    def get_intermediate_layers(self, pixel_values: torch.Tensor, *args, **kwargs):
        return self.model[0].get_intermediate_layers(self.to_input_dtype(pixel_values), *args, **kwargs)


class DetikzifyModel(LlamaModel):
    config_class = DetikzifyConfig

    def __init__(self, config: LlamaConfig):
        super(DetikzifyModel, self).__init__(config)

        if getattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            self.vision_model = DetikzifyVisionModel(create_vision_model(config.vision_tower))

    def initialize_vision_modules(
        self,
        vision_tower,
        patch_token_id,
        concat_patches=3,
        feature_layer=-1,
        modality_projector=None,
        **kwargs
    ):
        vision_model = create_vision_model(vision_tower, pretrained=True, **kwargs)
        self.vision_model = DetikzifyVisionModel(vision_model.to(self.device, self.dtype).eval().requires_grad_(False))

        processor = DetikzifyImageProcessor.from_pretrained(vision_tower)

        self.config.use_mm_proj = True
        self.config.vision_tower = vision_tower
        self.config.mm_hidden_size = vision_model.embed_dim * concat_patches
        self.config.patch_token_id = patch_token_id
        self.config.concat_patches = concat_patches
        self.config.feature_layer = int(clip(feature_layer, -(depth:=len(vision_model.blocks)), depth-1) % depth)
        self.config.vision_config = processor.to_dict() # type: ignore
        self.config.num_patches = vision_model.patch_embed.num_patches // concat_patches

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size,
                self.config.hidden_size,
                dtype=self.dtype,
                device=self.device
            )

        if modality_projector is not None:
            try: # first try to load as pickle
                mm_projector_weights = torch.load(modality_projector, map_location=self.device)
            except UnpicklingError: # and if that fails we try safetensors
                mm_projector_weights = load_file(modality_projector, device=str(self.device))
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return processor

    # https://stackoverflow.com/a/57208704
    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "vision_model"):
            self.set_vision_model = self.vision_model._apply(fn)
        return self

    def get_vision_features(self, pixel_values):
        concat, n_patch, layer = self.config.concat_patches, self.config.num_patches, self.config.feature_layer
        feats = self.vision_model.get_intermediate_layers(pixel_values, n=[layer], norm=True)[0]
        # in case the number of feature vectors is not divisible by the number
        # of patches we want to concatenate, we remove the first feature(s)
        return feats[:, -n_patch * concat:].reshape(-1, n_patch, feats.shape[-1] * concat)

    def is_tensor(self, thing):
        if isinstance(thing, (BatchEncoding, dict)):
            return all(isinstance(v, torch.Tensor) for v in thing.values())
        return isinstance(thing, torch.Tensor)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if hasattr(self, "vision_model") and (input_ids.shape[1] != 1 or self.training) and pixel_values is not None:
            with torch.no_grad():
                image_features = self.get_vision_features(pixel_values)
            image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(len(image_features[0]), self.config.mm_hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.config.image_token_id).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue

                cur_image_features = image_features[cur_image_idx].to(cur_input_embeds.device)
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == self.config.image_token_id).sum() != num_patches:
                    raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == self.config.image_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                cur_image_idx += 1

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(DetikzifyModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class DetikzifyForCausalLM(LlamaForCausalLM):
    config_class = DetikzifyConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = DetikzifyModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()


        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": kwargs.get("pixel_values", None),
            }
        )
        return model_inputs
