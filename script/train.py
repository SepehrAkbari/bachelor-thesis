"""
Trains a PPO agent for selecting S-pairs.
"""


import sys
import os
import torch
import torch.optim as optim
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
    )))

if not hasattr(np, 'string_'):
    np.string_ = np.bytes_

from src.include.wrapped import CLeadMonomialsEnv
from src.network import GeometricActorCritic
from src.ppo import PPOAgent, RolloutBuffer


def train(dist_name, pretrained_path, model_dir, log_dir,
          max_episodes=50000, max_timesteps=1000, update_timesteps=2048,
          lr=5e-5, gamma=0.99):
    
    writer = SummaryWriter(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    try:
        env = CLeadMonomialsEnv(ideal_dist=dist_name, k=1, elimination='gebauermoeller')
        print(f"Environment initialized for {dist_name}.")
    except Exception as e:
        print(f"Error: could not initialize environment ({e}).")
        return
    
    env.reset()
    sample_state = env.reset() # (Pairs, Feat)
    input_dim = sample_state.shape[1]
    print(f"State Dimension: {input_dim}")

    policy = GeometricActorCritic(input_dim, hidden_dim=128, pretrained_path=pretrained_path).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=lr)
    agent = PPOAgent(policy, optimizer, gamma=gamma)
    buffer = RolloutBuffer()
    
    time_step = 0
    
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        current_ep_reward = 0
        
        for t in range(max_timesteps):
            time_step += 1
            
            action, log_prob, val = agent.select_action(state)
            
            next_state, reward, done, _ = env.step(action)
            
            buffer.states.append(state)
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)
            buffer.values.append(val)
            
            state = next_state
            current_ep_reward += reward
            
            if time_step % update_timesteps == 0:
                agent.update(buffer)
            
            if done:
                break
        
        writer.add_scalar("Reward/Episode", current_ep_reward, i_episode)
        
        if i_episode % 10 == 0:
            print(f"Episode {i_episode} | Reward: {current_ep_reward:.2f} | Timesteps: {time_step}")
            
        # Save Model
        if i_episode % 100 == 0:
             torch.save(policy.state_dict(), f"{model_dir}/checkpoint_{i_episode}.pth")

    print("Training Finished.")
    writer.close()

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python train.py <DISTRIBUTION>")
        sys.exit()
        
    DIST_NAME = sys.argv[1]
    PRETRAINED_PATH = f"../model/pretrained/{DIST_NAME}/{DIST_NAME}.pth"
    
    MODEL_DIR = f"../model/ppo/{DIST_NAME}"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    LOG_DIR = f"{MODEL_DIR}/log/"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train(DIST_NAME, PRETRAINED_PATH, MODEL_DIR, LOG_DIR,
          max_episodes=50000, max_timesteps=1000, update_timesteps=2048,
          lr=5e-5, gamma=0.99)