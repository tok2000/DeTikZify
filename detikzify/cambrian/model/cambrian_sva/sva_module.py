import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialVisionAggregator(nn.Module):
    def __init__(self, kv_grid_shapes, kv_dim_list, hidden_dim, num_groups=3, grid_size=(27, 27), spatial_temp=1.0):
        super().__init__()
        self.num_groups = num_groups
        self.spatial_temp = spatial_temp
        self.hidden_dim = hidden_dim

        self.kv_grid_shapes = kv_grid_shapes
        self.query_h, self.query_w = grid_size
        self.total_queries = self.query_h * self.query_w # shapes the learned queries as a 2D grid
        self.query_group_split = self.total_queries // self.num_groups # number of query tokens inside one group
        
        self.learned_queries = nn.Parameter(torch.randn(1, self.total_queries, self.hidden_dim)) # learnable query tokens

        self.vision_projectors = nn.ModuleList([ # learnable projection for all vision features to match hidden_dim
            nn.Linear(kv_dim, hidden_dim) for kv_dim in kv_dim_list
        ])

        self.query_coords = torch.stack(torch.meshgrid( # used to compute distance-based attention bias
            torch.linspace(0, 1, self.query_h),
            torch.linspace(0, 1, self.query_w),
            indexing="ij" # create normalized (x,y) coords for each of the queries
        ), dim=-1).reshape(-1, 2)  # [HxW, 2]
        
    def forward(self, vision_latents_list, kv_coords_list=None):
        batch_size = vision_latents_list[0].shape[0]

        projected_tower_outputs = [ # each encoder output is projected to shared dim
            proj(x).to(dtype=self.vision_projectors[0].weight.dtype) # changed after 21475
            for proj, x in zip(self.vision_projectors, vision_latents_list)
        ]
        kv = torch.cat(projected_tower_outputs, dim=1)  # [batch_size, T_total, hidden_dim]

        if kv_coords_list is not None: # if positions are provided, use right away
            kv_coords = torch.cat(kv_coords_list, dim=1)  # [batch_size, T_total, 2]
        else: # assign positions to vision tokens
            # assume grid layout: normalize coords [0, 1] for spatial bias
            total_len = kv.shape[1]
            side = int(total_len ** 0.5) + 1
            grid = torch.stack(torch.meshgrid(
                torch.linspace(0, 1, side),
                torch.linspace(0, 1, side),
                indexing='ij'
            ), dim=-1).reshape(-1, 2).to(kv.device)
            kv_coords = grid[:total_len].unsqueeze(0).expand(batch_size, -1, -1)

        group_outputs = []
        for i in range(self.num_groups): # iterate through all groups
            num_rows, num_cols = self.query_h, self.query_w
            row_start = (i * num_rows) // self.num_groups
            row_end = ((i + 1) * num_rows) // self.num_groups

            start = row_start * num_cols
            end = row_end * num_cols

            q = self.learned_queries[:, start:end, :].expand(batch_size, -1, -1)
            q_coords = self.query_coords[start:end].unsqueeze(0).expand(batch_size, -1, -1).to(kv.device)

            kv_y_coords = kv_coords[:, :, 1]
            region_mask = (kv_y_coords >= row_start / num_rows) & (kv_y_coords < row_end / num_rows)
            region_mask = region_mask.unsqueeze(1).expand(-1, q.shape[1], -1)
            
            distance = torch.cdist(q_coords, kv_coords, p=2)  # use L2 distance between each query and vision token
            spatial_bias = -distance / self.spatial_temp  # compute spatial bias with negative and scaled for softmax
            spatial_bias = spatial_bias.masked_fill(~region_mask, -1e9) # mask out unrelated tokens in dynamic region-to-query alignment

            attn_scores = torch.matmul(q, kv.transpose(1, 2)) / (self.hidden_dim ** 0.5)
            attn_scores = attn_scores + spatial_bias # add spatial bias

            attn_weights = F.softmax(attn_scores, dim=-1).to(kv.dtype) # changed after 21469
            out = torch.bmm(attn_weights, kv) # batch matrix-matrix product
            group_outputs.append(out)

        return torch.cat(group_outputs, dim=1) # [batch_size, grid_size, hidden_dim]
        