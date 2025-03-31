import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads, attention_bias=False):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads

        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=attention_bias)
        self.k_proj = nn.Linear(kv_dim, hidden_dim, bias=attention_bias)
        self.v_proj = nn.Linear(kv_dim, hidden_dim, bias=attention_bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=attention_bias)

    def forward(self, vision_latents, queries, attention_mask=None):
        """
        Performs cross-attention between vision latents and queries.
        
        Args:
            vision_latents: Vision latents extracted from the vision encoder.
            queries: Latent tokens that attend to relevant features from the vision features.
            attention_mask: Mask to prevent attention to certain positions.
            
        Returns:
            Tensor: Refined queries after attending to vision features.
        """
        
        q = self.q_proj(queries)        
        k = self.k_proj(vision_latents)
        v = self.v_proj(vision_latents)

        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # scaled dot-product attention

        if attention_mask is not None: # mask out attention weights
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn_probs = F.softmax(attn_weights, dim=-1) # softmax over the last dimension
        attn_output = torch.matmul(attn_probs, v) # multiply attention weights with values
        
        batch_size, num_heads, num_queries, head_dim = attn_output.shape
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_queries, num_heads * head_dim)

        return self.out_proj(attn_output)

class AggregationBlock(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads):
        super().__init__()
        self.cross_attention = CrossAttention(q_dim, kv_dim, hidden_dim, num_heads) # apply cross-attention between vision latents and queries
        self.norm = nn.LayerNorm(hidden_dim) # apply layer normalization for stability and faster convergence
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), # feed-forward layer with 4 times the hidden dimension
            nn.GELU(), # GELU activation function
            nn.Linear(hidden_dim * 4, hidden_dim), # feed-forward layer with the hidden dimension
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim) # apply layer normalization for stability and faster convergence

    def forward(self, vision_latents, queries, attention_mask=None):
        queries = self.cross_attention(vision_latents, queries, attention_mask) + queries # Cross-attention update with residual connection
        queries = self.norm(queries) # apply layer normalization

        queries = self.ffn(queries) + queries # Feed-forward update with residual connection
        queries = self.norm_ffn(queries) # apply layer normalization

        return queries

class SpatialVisionAggregator(nn.Module):
    def __init__(self, q_dim, kv_dim_list, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.kv_dim_list = kv_dim_list

        self.aggregation_blocks = nn.ModuleList([
            AggregationBlock(q_dim, kv_dim_list[i], hidden_dim, num_heads) for i in range(num_layers)
        ])


    # queries: latent tokens that attend to relevant features from the vision features
    # vision_latents_attention_mask_list: raw image features extracted from multiple vision encoders
    def forward(self, queries, vision_latents_attention_mask_list):
        print("Number of Vision Towers Used:", len(vision_latents_attention_mask_list))
        
        for i, vision_latents in enumerate(vision_latents_attention_mask_list):
            print(f"Processing Vision Latents {i+1} of shape {vision_latents.shape}")
    
        for block, vision_latents in zip(self.aggregation_blocks, vision_latents_attention_mask_list):
            queries = block(vision_latents, queries)  # Cross-attention update
    
        return queries
