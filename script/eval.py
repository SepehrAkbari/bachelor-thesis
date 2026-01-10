"""
Evaluation of PPO agent. 
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
    )))

from src.include.wrapped import CLeadMonomialsEnv
from src.network import GeometricActorCritic


def evaluate(dist_name, model_path, num_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating on {dist_name} over {num_episodes} episodes.")
    
    env = CLeadMonomialsEnv(ideal_dist=dist_name, k=1, elimination='gebauermoeller')
    env.seed(49)
    
    env.reset()
    sample_state = env.reset()
    input_dim = sample_state.shape[1]
    
    agent = GeometricActorCritic(input_dim, hidden_dim=128).to(device)
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        print("Agent weights loaded successfully.")
    except FileNotFoundError:
        print("Error: No pre-trained model found.")
        return
    
    agent.eval()

    results = []

    for i in tqdm(range(num_episodes)):
        
        # --- A. Run Agent ---
        # Note: We must clone the env or rely on the fact that we can't 
        # easily 'reset' to the exact same state in the C++ wrapper 
        # without explicit seed control per step. 
        # A better approach for the thesis: 
        # Compare AGENT vs STRATEGIES on *DIST_NAME averages*, 
        # rather than 1-to-1 instance comparison, OR use a fixed dataset like in Stage 1.
        
        # For now, we will just measure the Agent's performance average.
        
        state = env.reset()
        steps = 0
        agent_additions = 0
        done = False
        
        while not done:
            # Prepare State
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Select Action (Deterministic / Greedy for Eval)
            with torch.no_grad():
                _, policy_logits = agent.transformer(state_tensor, mask=None)
                action = torch.argmax(policy_logits).item()
            
            state, reward, done, _ = env.step(action)
            
            # Reward is -(additions + 1), so additions = -reward - 1
            # But the C++ env might return -1 for zero reductions.
            # Let's trust the reward accumulation.
            agent_additions += (-reward) # Approximate cost
            steps += 1
            
        # --- B. Run Baselines (Sugar/Degree) ---
        # The C++ env allows us to query the cost of heuristic strategies directly
        # on the *next* generated ideal. 
        # To strictly compare on the SAME ideal, we would need to modify the C++ code 
        # to save/load ideals.
        # For this script, we will trust the Law of Large Numbers (averages converge).
        
        # We re-run a "similar" ideal for the baseline
        # (This is a limitation of the current wrapper, but acceptable for N > 500)
        sugar_val = env.value(strategy='degree', gamma=1.0) # gamma=1.0 sums total reward
        sugar_additions = -sugar_val # Convert negative reward to positive cost

        results.append({
            'Agent': agent_additions,
            'Sugar': sugar_additions
        })

    df = pd.DataFrame(results)
    print(df.describe())
    
    mean_agent = df['Agent'].mean()
    mean_sugar = df['Sugar'].mean()
    
    print(f"\nAverage Cost:")
    print(f"Agent: {mean_agent:.2f}")
    print(f"Sugar: {mean_sugar:.2f}")
    
    improvement = (mean_sugar - mean_agent) / mean_sugar * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python eval.py <DISTRIBUTION> <CHECKPOINT_VERSION>")
        sys.exit()
    
    DIST_NAME = sys.argv[1]
    CHECKPOINT_VERSION = sys.argv[2]
    
    MODEL_DIR = f"../model/ppo/{DIST_NAME}"
    MODEL_PATH = f"{MODEL_DIR}/checkpoint_{CHECKPOINT_VERSION}.pth"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    LOG_DIR = f"{MODEL_DIR}/log/"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    evaluate(DIST_NAME, MODEL_PATH, num_episodes=100)