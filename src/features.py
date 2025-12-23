import torch

def extract_features(ideal_tensor):
    """

    """
    # Sum of exponents per generator (degrees)
    degrees = ideal_tensor.sum(dim=-1)
    
    max_deg = torch.max(degrees)
    min_deg = torch.min(degrees)
    mean_deg = torch.mean(degrees)
    std_deg = torch.std(degrees)
    
    # Sparsity (count of zero elements)
    sparsity = (ideal_tensor == 0).float().mean()
    
    # Variable Support (how many unique variables are used)
    var_support = (ideal_tensor.sum(dim=0) > 0).float().sum()
    
    return torch.stack([max_deg, min_deg, mean_deg, std_deg, sparsity, var_support])