"""
Actor-Critic network.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import os

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
    )))

from .model import SetTransformer


class GeometricActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, pretrained_path=None):
        super(GeometricActorCritic, self).__init__()
        
        self.transformer = SetTransformer(input_dim=input_dim, hidden_dim=hidden_dim)
        
        if pretrained_path:
            try:
                state_dict = torch.load(pretrained_path)
                self.transformer.load_state_dict(state_dict, strict=False)
                print(f"Pre-trained weights loaded ({pretrained_path}).")
            except Exception as e:
                print(f"Warning: Could not load weights ({e}).")

    def forward(self, x, mask=None):
        """
        Returns:
            pi: Action distribution (Categorical)
            v:  Value estimate (Scalar)
        """
        # value_pred: (B, 1), policy_logits: (B, P)
        value_pred, policy_logits = self.transformer(x, mask)
        
        if mask is not None:
            policy_logits = policy_logits.masked_fill(mask, -1e9)
            
        dist = Categorical(logits=policy_logits)
        return dist, value_pred