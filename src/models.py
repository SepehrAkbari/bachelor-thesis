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
        # x shape: (batch, num_generators, feature_dim)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class BuchbergerTransformer(nn.Module):
    def __init__(self, input_dim=6, num_generators=10, d_model=64, n_heads=4):
        super().__init__()
        # individual exponent vectors to embedding space
        self.embedding = nn.Linear(input_dim, d_model)
        
        self.block1 = SetAttentionBlock(d_model, n_heads)
        self.block2 = SetAttentionBlock(d_model, n_heads)
        
        # Pooling layer to aggregate generator features
        self.pool = nn.AdaptiveAvgPool1d(1) 
        
        # Regression head to predict additions
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, num_gens, exp_dim) -> (batch, 10, 6)
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        
        # Pool across the "generator" dimension (dim=1)
        x = x.transpose(1, 2) # (batch, d_model, num_gens)
        x = self.pool(x).squeeze(-1) # (batch, d_model)
        
        return self.head(x).squeeze(-1)