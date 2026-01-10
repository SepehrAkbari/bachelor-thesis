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
        # Embed each pair independently
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