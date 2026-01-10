"""
Proximal Policy Optimization (PPO) agent.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class PPOAgent:
    def __init__(self, policy, optimizer, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.mse_loss = nn.MSELoss()
        self.device = next(policy.parameters()).device

    def select_action(self, state_matrix):
        """
        Selects an action for a single environment step.
        State: numpy array (Pairs, Features)
        """
        with torch.no_grad():
            # Added batch dimension: (1, Pairs, Feat)
            state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
            
            dist, value = self.policy(state_tensor, mask=None)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()

    def update(self, buffer):
        """
        Performs PPO update using collected buffer.
        """
        states = [torch.FloatTensor(s) for s in buffer.states]
        actions = torch.LongTensor(buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(buffer.log_probs).to(self.device)
        rewards = buffer.rewards
        is_terminals = buffer.is_terminals
        old_values = buffer.values

        # Pad states: (Batch, Max_Pairs, Feat)
        states_padded = pad_sequence(states, batch_first=True, padding_value=0).to(self.device)
        
        lengths = torch.tensor([len(s) for s in states])
        max_len = states_padded.size(1)
        mask = torch.arange(max_len)[None, :] >= lengths[:, None]
        mask = mask.to(self.device)

        # Monte Carlo estimate of returns & advantages (GAE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(old_values, dtype=torch.float32).to(self.device)
        
        returns = []
        advantages = []
        discounted_reward = 0
        gae = 0
        lam = 0.95 # GAE lambda
        
        for i in reversed(range(len(rewards))):
            if is_terminals[i]:
                discounted_reward = 0
                gae = 0
                next_val = 0
            else:
                next_val = old_values[i+1] if i+1 < len(old_values) else 0
                
            delta = rewards[i] + self.gamma * next_val - old_values[i]
            gae = delta + self.gamma * lam * gae
            advantages.insert(0, gae)
            
            discounted_reward = rewards[i] + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.k_epochs):
            dist, state_values = self.policy(states_padded, mask)
            
            action_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = state_values.squeeze()
            
            ratios = torch.exp(action_logprobs - old_log_probs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            
            self.optimizer.step()
            
        buffer.clear()

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]