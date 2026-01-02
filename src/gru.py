import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # x shape: [batch, max_gens, features]
        
        # Pack the sequence
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        _, hn = self.gru(x_packed)
        
        # hn shape: [num_layers, batch, hidden_dim]
        features = hn[-1] 
        
        # Regression head
        out = self.fc(features)
        return out