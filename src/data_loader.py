import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class GroebnerDataset(Dataset):
    def __init__(self, folder_path, dataset_name):
        self.name = dataset_name
        stats_dir = os.path.join(folder_path, 'stats', dataset_name)
        
        # Polynomial additions
        degree_df = pd.read_csv(os.path.join(stats_dir, f"{dataset_name}_degree.csv"))
        self.targets = torch.tensor(degree_df['PolynomialAdditions'].values, dtype=torch.float32)
        
        # Ideals as exponent vectors
        self.inputs = np.load(os.path.join(stats_dir, f"{dataset_name}.npy"))
        
        # Normalization
        self.target_mean = self.targets.mean()
        self.target_std = self.targets.std()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Original shape is (NumSamples, NumGenerators, ExponentVectorSize)
        # We return (NumGenerators, ExponentVectorSize) to treat as a 'Set'
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = self.targets[idx]
        return x, y

def get_data_loader(folder_path, dataset_name, batch_size=128, shuffle=True):
    dataset = GroebnerDataset(folder_path, dataset_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)