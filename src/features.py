import numpy as np

import torch


def extract_feats(data_loader, n_vars):
    """
    1. Min Degree
    2. Max Degree
    3. Mean Degree
    4. Std Degree
    5. Number of Pure Power Leading Terms
    + Num_Generators for Toric
    """
    X_features = []
    y_targets = []
    
    for batch_X, batch_y, batch_lens in data_loader:
        # batch_X shape: [Batch, Max_Gens, Padded_Feats]
        
        batch_size = batch_X.shape[0]
        
        for i in range(batch_size):
            num_gens = batch_lens[i]
            gens = batch_X[i, :num_gens, :] # Shape: [Num_Gens, 16]
            
            # Computing degrees
            # feature vector: [Term1_vars, Term2_vars, Padding...]
            term1_exps = gens[:, :n_vars]
            term2_exps = gens[:, n_vars:2*n_vars]
            
            deg_term1 = term1_exps.sum(dim=1)
            deg_term2 = term2_exps.sum(dim=1)
            
            gen_degrees = torch.maximum(deg_term1, deg_term2).numpy()
            
            # Computing statistics
            min_deg = np.min(gen_degrees)
            max_deg = np.max(gen_degrees)
            mean_deg = np.mean(gen_degrees)
            std_deg = np.std(gen_degrees)
            
            # Computing Pure Powers (on Leading Term / Term 1)
            # (monomial is a pure power iff one variable has a non-zero exponent)
            non_zero_counts = (term1_exps > 0).sum(dim=1)
            num_pure_powers = (non_zero_counts == 1).sum().item()
            
            # Combining
            features = [min_deg, max_deg, mean_deg, std_deg, num_pure_powers, num_gens.item()]
            
            X_features.append(features)
            y_targets.append(batch_y[i].item())
            
    return np.array(X_features), np.array(y_targets)