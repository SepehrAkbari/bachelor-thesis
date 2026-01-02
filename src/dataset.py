import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


DATA_ROOT = "../data/groebner_dataset"


class GroebnerDataset(Dataset):
    def __init__(self, dist_name: str, split: str = 'train', root_dir: str = DATA_ROOT, padding: int = 16):
        """
        Args:
            dist_name (str): '3-20-10-weighted', 'toric-6-0-5-8', etc.
            split (str): 'train', 'val', or 'test' (80/10/10)
            root_dir (str): Path to the dataset root
            padding (int): For universal training, we need consistent feature dims. 
                            Binomial=6 (3vars*2), Toric=16 (8vars*2). 
                            padding=16 to pad binomials with zeros.
        """
        self.dist_name = dist_name
        self.stats_dir = os.path.join(root_dir, 'stats', dist_name)
        
        # Features (X)
        npy_path = os.path.join(self.stats_dir, f"{dist_name}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Could not find feature file: {npy_path}")
            
        raw_X = np.load(npy_path, allow_pickle=True)
        
        # Targets (y)
        csv_path = os.path.join(self.stats_dir, f"{dist_name}_degree.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find target file: {csv_path}")
            
        raw_y = pd.read_csv(csv_path)['PolynomialAdditions'].to_numpy()

        # Splitting (80/10/10)
        total_len = len(raw_X)
        train_len = int(0.8 * total_len)
        valid_len = int(0.1 * total_len)
        
        if split == 'train':
            self.X = raw_X[:train_len]
            self.y = raw_y[:train_len]
        elif split == 'val':
            self.X = raw_X[train_len : train_len + valid_len]
            self.y = raw_y[train_len : train_len + valid_len]
        elif split == 'test':
            self.X = raw_X[train_len + valid_len:]
            self.y = raw_y[train_len + valid_len:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.pad_dim = padding

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # features shape: (num_generators, input_vars * 2)
        features = self.X[idx] 
        target = self.y[idx]

        # Tensor conversion
        if features.dtype == np.object_:
            features = np.array(features.tolist()).astype(np.float32)
        else:
            features = features.astype(np.float32)
            
        features_tensor = torch.from_numpy(features)
        
        # features are dim 6 (Binomial) but we want 16 (Toric max), we pad the last dimension.
        current_dim = features_tensor.shape[1]
        if self.pad_dim and current_dim < self.pad_dim:
            # Pad format: (padding_left, padding_right, padding_top, padding_bottom...)
            pad_amt = self.pad_dim - current_dim
            features_tensor = torch.nn.functional.pad(features_tensor, (0, pad_amt))

        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        return features_tensor, target_tensor
    
    
def pad_collate(batch):
    """
    Pads the sequence of generators to the max number of generators in the batch.
    """
    (xx, yy) = zip(*batch)
    
    # output: (Batch, Max_Num_Gens, Features)
    x_padded = pad_sequence(xx, batch_first=True, padding_value=0)
    
    # 1 where real data exists, 0 where padded
    lengths = torch.tensor([x.shape[0] for x in xx])
    
    y_stacked = torch.stack(yy)
    
    return x_padded, y_stacked, lengths