"""
Dataset of polynomial ideals in tensor format.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import sympy as sp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
    )))

from src.buchberger import LeadMonomialsEnv
from src.ideal import FixedIdealGenerator


def parse_ideal(ideal_str, ring_vars):
    """
    Parses an ideal e.g. "{x^2+y,z}", into list of SymPy polynomials using ring variables.
    """
    if not isinstance(ideal_str, str):
        return []

    clean_str = (ideal_str.replace('{', '[')
                          .replace('}', ']')
                          .replace('^', '**')
                          .replace('|', ','))
    
    eval_ctx = {str(var): var for var in ring_vars}
    
    try:
        poly_list = eval(clean_str, {}, eval_ctx)
        return poly_list
    except Exception as e:
        print(f"Error: cannot parse ideal string {ideal_str} ({e})")
        raise e

def process_dataset(dist_name, data_dir, output_path):
    """
    Processes the dataset for a given distribution from CSV to tensor format.
    """
    ideals_path = os.path.join(data_dir, f"{dist_name}.csv")
    labels_path = os.path.join(data_dir, f"{dist_name}_degree.csv")
    
    print(f"Processing {dist_name}...")
    
    try:
        ideals_df = pd.read_csv(ideals_path)
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError as e:
        print(f"Error: Missing file ({e}).")
        return

    if len(ideals_df) != len(labels_df):
        print(f"Warning: Mismatch in length. Ideals: {len(ideals_df)}, Labels: {len(labels_df)}")
        min_len = min(len(ideals_df), len(labels_df))
        ideals_df = ideals_df.iloc[:min_len]
        labels_df = labels_df.iloc[:min_len]
    
    try:
        n_vars = int(dist_name.split('-')[0])
    except ValueError:
        print("Error: Unable to parse number of variables from distribution name. Expeting e.g. '3-20-4-uniform'.")
        exit()

    var_names = [chr(i) for i in range(ord('a'), ord('a') + n_vars)]
    ring, *variables = sp.ring(",".join(var_names), sp.FF(32003), 'grevlex')
    
    env = LeadMonomialsEnv(ideal_dist=dist_name, k=1, elimination='gebauermoeller')
    
    processed_data = []
    
    for idx in tqdm(range(len(ideals_df))):
        row_ideal = ideals_df.iloc[idx]
        row_label = labels_df.iloc[idx]
        
        # Parsing
        try:
            polys = parse_ideal(row_ideal['Ideal'], variables)
            if not polys: continue
        except Exception as e:
            print(f"Warning: Skipping row {idx} due to parsing error ({e}).")
            continue
        
        # Generating S-pairs matrix
        try:
            env.env.ideal_gen = FixedIdealGenerator(polys)
            state_matrix = env.reset()
            
            # Nodes shape: (N_poly, FeatureDim)
            poly_features = np.array(env.leads, dtype=np.float32)
            # Edges shape: (N_pair, 2)
            pair_indices = np.array(env.env.P, dtype=np.int64)
        except Exception as e:
            print(f"Warning: Skipping row {idx} ({e}).")
            continue

        # Label extraction
        raw_additions = row_label['PolynomialAdditions']
        label = np.log1p(raw_additions)
        
        # Compiling
        processed_data.append({
            'features': torch.tensor(state_matrix, dtype=torch.float32),
            'poly_features': torch.tensor(poly_features, dtype=torch.float32),
            'pair_indices': torch.tensor(pair_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
            'raw_additions': raw_additions,
            'original_index': idx
        })
    
    print(f"{len(processed_data)} samples processed.")
    torch.save(processed_data, output_path)
    

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python process_data.py <DISTRIBUTION>")
        sys.exit()
        
    DISTRIBUTION = sys.argv[1]
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "stats", DISTRIBUTION)
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "tensors")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{DISTRIBUTION}.pt")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory ({DATA_DIR}) not found.")
    else:
        process_dataset(DISTRIBUTION, DATA_DIR, OUTPUT_FILE)