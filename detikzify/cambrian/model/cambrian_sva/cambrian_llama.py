import torch
import torch.nn as nn
from transformers import LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast

def print_mem(tag=""): # added after 19528
    if torch.cuda.is_available():
        print(f"\n [MEM] {tag}")
        allocated = torch.cuda.memory_allocated() / 1024**2
        if isinstance(allocated, (int, float)):
            print(f"→ Allocated: {allocated:.2f} MB")
        else:
            print(f"→ Allocated: {allocated} MB (not a float)")
        print(f"→ Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"→ Max Alloc: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"→ Max Resrv: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    


class VisionCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.resid_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_hidden, vision_latents, attn_mask=None):
        vision_latents = vision_latents.to(dtype=text_hidden.dtype, device=text_hidden.device) # added after 21477
        self.norm_q = self.norm_q.to(dtype=vision_latents.dtype, device=vision_latents.device) # added after 21472
        q = self.norm_q(text_hidden) # apply norm on queries
        self.norm_kv = self.norm_kv.to(dtype=vision_latents.dtype, device=vision_latents.device) # added after 21472
        kv = self.norm_kv(vision_latents) # apply norm on key-values

        attn_output, _ = self.cross_attn(q, kv, kv, key_padding_mask=attn_mask) # cross attention
        attn_output = self.dropout(attn_output)
        return text_hidden + self.resid_proj(attn_output)


class DetikzifyCambrianLlamaModel(nn.Module):
    def __init__(
        self, 
        base_llama: LlamaModel, 
        vision_fusion_layers=[0, 3, 6],
        vision_dim=None,
        max_vision_tokens=300, 
        sva_blocks=None
    ):
        super().__init__()
        self.llama = base_llama # LLaMA base model for basic functionality
        self.vision_fusion_layer_index = vision_fusion_layers # layers where SVA outputs are injected
        self.hidden_size = base_llama.config.hidden_size # LLaMA model hidden size
        #self.vision_dim = vision_dim or self.hidden_size
        self.num_heads = base_llama.config.num_attention_heads # number of attention heads in cross attention
        self.vision_dim = vision_dim or self.hidden_size

        self.cross_attn_blocks = nn.ModuleDict({ # for each fusion layer, add a cross-attention block which learns how to merge text and image features
            str(i): VisionCrossAttentionBlock(self.hidden_size, self.num_heads)
            for i in vision_fusion_layers
        })

        self.fusion_gates = nn.ParameterDict({ # add gated vision fusion
            str(i): nn.Parameter(torch.tensor(1.0))
            for i in self.vision_fusion_layer_index
        })

        self.vision_pos_embeddings = None
        self.sva_blocks = sva_blocks

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        vision_latents_list=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.llama.embed_tokens(input_ids) # prepare embeddings

        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [] if output_attentions else None
        next_decoder_cache = [] if use_cache else None

        if position_ids is None:
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)
        
        for idx, decoder_layer in enumerate(self.llama.layers): # iterate through Llama layers
            #print_mem(f"Before Llama layer {idx}") # added after 19528, deleted after 21513
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if idx in self.vision_fusion_layer_index and vision_latents_list is not None: # inject vision features at designated layers
                sva_block_idx = self.vision_fusion_layer_index.index(idx)
                #print_mem(f"Before SVA Block {sva_block_idx}") # added after 19528, deleted after 21513
                vision_latents = self.sva_blocks[sva_block_idx](vision_latents_list).to(dtype=hidden_states.dtype, device=hidden_states.device) # changed after 21476
                #print_mem(f"After SVA Block {sva_block_idx}") # added after 19528, deleted after 21536
            
                if self.vision_pos_embeddings is None or self.vision_pos_embeddings.size(1) < vision_latents.size(1):
                    self.vision_pos_embeddings = nn.Parameter(
                        torch.randn(1, vision_latents.size(1), vision_latents.size(2), device=vision_latents.device)
                    )
                vision_latents = vision_latents + self.vision_pos_embeddings[:, :vision_latents.size(1), :]
                
                vision_out = self.cross_attn_blocks[str(idx)](hidden_states, vision_latents) # call VisionCrossAttentionBlock
                hidden_states = hidden_states + self.fusion_gates[str(idx)] * vision_out # call gated vision fusion

            if attention_mask is not None and attention_mask.dim() == 2:
                # [batch, seq] -> [batch, 1, tgt_seq, src_seq]
                attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            
            layer_outputs = decoder_layer( # regular Llama decoder layer
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value = past_key_values[idx] if past_key_values is not None and idx < len(past_key_values) else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache.append(layer_outputs[1])
            if output_attentions:
                all_self_attns.append(layer_outputs[1])

            #print_mem(f"After Llama layer {idx}") # added after 19528, deleted after 21536

        hidden_states = self.llama.norm(hidden_states) # add norm

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        if torch.cuda.is_available():
            pass #print(torch.cuda.memory_summary()) # deleted after 21536

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(next_decoder_cache) if use_cache else None,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_self_attns) if output_attentions else None,
        )

    def get_input_embeddings(self):
        return self.llama.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.llama.set_input_embeddings(value)

    @property
    def gradient_checkpointing(self):
        return getattr(self.llama, "gradient_checkpointing", False)
    
    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        if hasattr(self.llama, "gradient_checkpointing"):
            self.llama.gradient_checkpointing = value
