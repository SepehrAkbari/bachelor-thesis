'''
Dataset of polynomial ideals in tensor format
'''

import sys
import os
import pandas as pd
import numpy as np
import torch
import sympy as sp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from buchberger import LeadMonomialsEnv
from ideals import FixedIdealGenerator


def parse_ideal(ideal_str, ring_vars):
    """
    Parses an ideal e.g. "{x^2 + y, z}", into list of SymPy polynomials using ring variables.
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
        print(f"Error parsing ideal: {ideal_str}")
        raise e

def process_dataset(dist_name, data_dir, output_path):
    """
    Processes the dataset for a given distribution from CSV to tensor format.
    """
    # 1. Construct File Paths
    # Input file: contains the 'Ideal' column
    ideals_path = os.path.join(data_dir, f"{dist_name}.csv")
    # Label file: contains 'PolynomialAdditions' from the 'degree' strategy
    labels_path = os.path.join(data_dir, f"{dist_name}_degree.csv")
    
    print(f"Processing {dist_name}...")
    print(f"  - Ideals: {ideals_path}")
    print(f"  - Labels: {labels_path}")
    
    # 2. Load Dataframes
    try:
        ideals_df = pd.read_csv(ideals_path)
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError as e:
        print(f"Error: Missing file. {e}")
        return

    # 3. Validation
    if len(ideals_df) != len(labels_df):
        print(f"Warning: Mismatch in length! Ideals: {len(ideals_df)}, Labels: {len(labels_df)}")
        # We process the intersection of rows to be safe
        min_len = min(len(ideals_df), len(labels_df))
        ideals_df = ideals_df.iloc[:min_len]
        labels_df = labels_df.iloc[:min_len]
    
    # 4. Ring Setup
    # Assumes format like '3-20-4-uniform-test' -> 3 variables
    try:
        n_vars = int(dist_name.split('-')[0])
    except ValueError:
        # Fallback if naming convention is different, standard is usually 3
        print("Could not infer variables from name, defaulting to 3.")
        n_vars = 3

    var_names = [chr(i) for i in range(ord('a'), ord('a') + n_vars)]
    # Create ring with 32003 characteristic (standard for this dataset)
    ring, *variables = sp.ring(",".join(var_names), sp.FF(32003), 'grevlex')
    
    # 5. Environment Setup
    # k=1 extracts lead monomials for the pair (f_i, f_j)
    # elimination='gebauermoeller' matches the standard reduction strategy
    env = LeadMonomialsEnv(ideal_dist=dist_name, k=1, elimination='gebauermoeller')
    
    processed_data = []
    
    # 6. Iteration
    # We iterate through both dataframes simultaneously
    for idx in tqdm(range(len(ideals_df))):
        row_ideal = ideals_df.iloc[idx]
        row_label = labels_df.iloc[idx]
        
        # A. Parse Ideal
        try:
            polys = parse_ideal(row_ideal['Ideal'], variables)
            if not polys: continue
        except Exception as e:
            print(f"Skipping row {idx} due to parsing error: {e}")
            continue
        
        # B. Generate S-Pairs Matrix (The "Geometric" Input)
        try:
            # Inject specific ideal into environment
            env.env.ideal_gen = FixedIdealGenerator(polys)
            # Reset triggers the first update() and returns the pair matrix
            state_matrix = env.reset()
        except Exception as e:
             # Sometimes an ideal is already a Groebner basis (empty pairs), skip these
            # print(f"Skipping row {idx} (Likely 0 pairs): {e}")
            continue

        # C. Get Label
        # We use log1p(additions) because the values span orders of magnitude
        raw_additions = row_label['PolynomialAdditions']
        label = np.log1p(raw_additions)
        
        # D. Save
        processed_data.append({
            'features': torch.tensor(state_matrix, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'raw_additions': raw_additions,
            'original_index': idx
        })
    
    # 7. Save Output
    print(f"Successfully processed {len(processed_data)} samples.")
    print(f"Saving to {output_path}")
    torch.save(processed_data, output_path)
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_data.py <DISTRIBUTION>")
        # Example: python process_data.py 3-20-4-uniform-test
        sys.exit()
        
    DISTRIBUTION = sys.argv[1]
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "groebner_dataset", "stats", DISTRIBUTION)
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "groebner_dataset", "tensors")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{DISTRIBUTION}.pt")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
    else:
        process_dataset(DISTRIBUTION, DATA_DIR, OUTPUT_FILE)