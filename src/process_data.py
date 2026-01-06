import sys
import os
import pandas as pd
import numpy as np
import torch
import sympy as sp
from tqdm import tqdm

# Add the src directory to the path so we can import the copied modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from buchberger import LeadMonomialsEnv
from ideals import FixedIdealGenerator

def parse_ideal_string(ideal_str, ring_vars):
    """
    Parses a string representation of an ideal like "{x^2 + y, z}" 
    into a list of SymPy polynomials using the provided ring variables.
    """
    # Clean the string to match Python syntax
    clean_str = (ideal_str.replace('{', '[')
                          .replace('}', ']')
                          .replace('^', '**')
                          .replace('|', ',')) # Handle cases where | is used as separator
    
    # We need to evaluate this string in a context where x, y, z, etc. are defined
    # Create a local dictionary for eval context
    eval_ctx = {str(var): var for var in ring_vars}
    
    try:
        poly_list = eval(clean_str, {}, eval_ctx)
        return poly_list
    except Exception as e:
        print(f"Error parsing ideal: {ideal_str}")
        raise e

def process_dataset(dist_name, csv_path, output_path):
    print(f"Processing {dist_name} from {csv_path}...")
    
    # 1. Load the CSV
    df = pd.read_csv(csv_path)
    
    # 2. Setup the Ring (Logic adapted from deepgroebner scripts)
    # Assumes format like '3-20-10-uniform' where first number is num_vars
    n_vars = int(dist_name.split('-')[0])
    var_names = [chr(i) for i in range(ord('a'), ord('a') + n_vars)] # a, b, c...
    
    # Create the ring and variables. Using same field and order as deepgroebner default.
    ring, *variables = sp.ring(",".join(var_names), sp.FF(32003), 'grevlex')
    
    # 3. Initialize the Environment
    # We use LeadMonomialsEnv because it automatically converts pairs to a matrix 
    # of exponents (which is exactly what a Neural Network needs).
    # k=1 gives us the lead monomial of f_i and f_j for every pair.
    env = LeadMonomialsEnv(ideal_dist=None, k=1, elimination='gebauermoeller')
    
    processed_data = []
    
    print("Converting ideals to S-pair matrices...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # A. Parse the Ideal
        try:
            polys = parse_ideal_string(row['Ideal'], variables)
        except:
            continue

        # B. Inject into Environment
        # We use FixedIdealGenerator to force the env to use THIS specific ideal
        env.env.ideal_gen = FixedIdealGenerator(polys)
        
        # C. Reset to generate the initial S-pairs
        # The env.reset() in LeadMonomialsEnv returns the matrix representation
        state_matrix = env.reset()
        
        # D. Get the Label (Log of additions is usually more stable for regression)
        # Using log1p to handle 0 cleanly, though additions should be >0
        label = np.log1p(row['PolynomialAdditions']) 
        
        # E. Store as Tensors
        # Input: [NumPairs, FeatureDim] (Float for NN, though data is Int)
        # Label: [1]
        processed_data.append({
            'features': torch.tensor(state_matrix, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'original_index': idx
        })
        
    print(f"Saving {len(processed_data)} samples to {output_path}...")
    torch.save(processed_data, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: train.py <DISTRIBUTION>")
        sys.exit()
    DISTRIBUTION = sys.argv[1].lower()
    # CONFIGURATION
    # Change these paths to match your folder structure
    DATA_DIR = "../data/groebner_dataset"
    CSV_FILE = f"{DATA_DIR}/stats/{DISTRIBUTION}/{DISTRIBUTION}.csv" 
    OUTPUT_FILE = f"{DATA_DIR}/{DISTRIBUTION}/{DISTRIBUTION}.pt"
    
    # Ensure data directory exists
    os.makedirs(f"{DATA_DIR}/{DISTRIBUTION}", exist_ok=True)
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: Could not find {CSV_FILE}.")
    else:
        process_dataset(DISTRIBUTION, CSV_FILE, OUTPUT_FILE)