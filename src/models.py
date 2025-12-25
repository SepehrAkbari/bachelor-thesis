import torch
import torch.nn as nn
import torch.nn.functional as F

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class BuchbergerTransformer(nn.Module):
    def __init__(self, input_dim=8, d_model=64, n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.block1 = SetAttentionBlock(d_model, n_heads)
        self.block2 = SetAttentionBlock(d_model, n_heads)
        
        self.pool = nn.AdaptiveAvgPool1d(1) 
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, num_gens, current_dim)
        batch_size, num_gens, current_dim = x.shape
        
        # If input is Binomial (dim 6) but model expects Toric (dim 8), pad with zeros.
        if current_dim < self.input_dim:
            padding = self.input_dim - current_dim
            # Pad last dimension: (padding_left, padding_right)
            x = F.pad(x, (0, padding)) 
            
        x = self.embedding(x)   
        x = self.block1(x)
        x = self.block2(x)
        
        # Global Pooling: (batch, num_gens, d_model) -> (batch, d_model)
        x = x.transpose(1, 2) 
        x = self.pool(x).squeeze(-1) 
        
        return self.head(x).squeeze(-1)