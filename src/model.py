"""
Model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(SetTransformer, self).__init__()
        
        # Embedding Layer: Projects raw pair features to latent space
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer Encoder: Processes the set of pairs
        # input: (Batch, Num_Pairs, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        # Value Head: Predicts total complexity
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Policy Head: Predicts score for each pair
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (Batch, Max_Pairs, Input_Dim)
            mask: BoolTensor of shape (Batch, Max_Pairs). True where padding exists.
        """
        # x shape: (B, P, Input_Dim) -> (B, P, Hidden)
        h = self.embedding(x)
        
        # Inter-pair communication via Self-Attention
        # mask is passed to ignore padding tokens
        h_transformed = self.encoder(h, src_key_padding_mask=mask)
        
        # Global Average Pooling
        # must mask out the padding before averaging
        if mask is not None:
            input_mask = (~mask).unsqueeze(-1).float() # (B, P, 1)
            sum_h = (h_transformed * input_mask).sum(dim=1)
            count = input_mask.sum(dim=1).clamp(min=1e-9)
            global_feat = sum_h / count
        else:
            global_feat = h_transformed.mean(dim=1)
            
        value_pred = self.value_head(global_feat) # Scalar complexity score
        
        # Score for every pair
        policy_logits = self.policy_head(h_transformed).squeeze(-1) # (B, P)
        
        if mask is not None:
            # Mask out padding logits so they don't affect softmax
            policy_logits = policy_logits.masked_fill(mask, -1e9)
            
        return value_pred, policy_logits
    
class BGNNLayer(nn.Module):
    def __init__(self, poly_dim, pair_dim, hidden_dim):
        super(BGNNLayer, self).__init__()
    
        self.message_mlp = nn.Sequential(
            nn.Linear(pair_dim + 2 * poly_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.pair_update = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, poly_feats, pair_feats, pair_indices):
        """
        poly_feats: (Batch, Num_Polys, Poly_Dim)
        pair_feats: (Batch, Num_Pairs, Pair_Dim)
        pair_indices: (Batch, Num_Pairs, 2) -> Indices of (i, j) for each pair
        """
        batch_size, num_pairs, _ = pair_feats.shape
        
        # Expand: (B, P, 2) -> (B, P, 2, D)
        idx = pair_indices.unsqueeze(-1).expand(-1, -1, -1, poly_feats.size(-1))
        
        # Gather: parents shape is (B, P, 2, Poly_Dim)
        parents = torch.gather(poly_feats, 1, idx)
        
        # Flatten parents: (B, P, 2 * Poly_Dim)
        parents_flat = parents.view(batch_size, num_pairs, -1)
        
        # Concatenate: [Pair_Feat || Poly_i || Poly_j]
        combined = torch.cat([pair_feats, parents_flat], dim=-1)
        
        messages = self.message_mlp(combined)
        
        out = self.pair_update(messages)
        return out

class BGNN(nn.Module):
    def __init__(self, poly_input_dim, pair_input_dim, hidden_dim=64, num_layers=2):
        super(BGNN, self).__init__()
        
        self.poly_embed = nn.Linear(poly_input_dim, hidden_dim)
        self.pair_embed = nn.Linear(pair_input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            BGNNLayer(hidden_dim, hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # Policy head (Actor): score per pair
        self.policy_head = nn.Linear(hidden_dim, 1)
        
        # Value head (Critic): score for whole set
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, poly_x, pair_x, pair_indices, mask=None):
        """
        poly_x: (B, N_poly, F_poly)
        pair_x: (B, N_pair, F_pair)
        pair_indices: (B, N_pair, 2)
        mask: (B, N_pair) -> True if padding
        """
        h_poly = self.poly_embed(poly_x)
        h_pair = self.pair_embed(pair_x)
        
        for layer in self.layers:
            h_pair = layer(h_poly, h_pair, pair_indices) + h_pair # Residual
            
        # Heads
        logits = self.policy_head(h_pair).squeeze(-1)
        
        # Global pooling for value
        if mask is not None:
            input_mask = (~mask).unsqueeze(-1).float()
            sum_h = (h_pair * input_mask).sum(dim=1)
            count = input_mask.sum(dim=1).clamp(min=1e-9)
            global_feat = sum_h / count
        else:
            global_feat = h_pair.mean(dim=1)
            
        value = self.value_head(global_feat)
        
        return value, logits