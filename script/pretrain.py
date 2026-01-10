"""
Trains a network to predict log-additions.
"""

import sys
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
    )))

from src.model import SetTransformer


class GroebnerDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path, weights_only=False)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]['features'], self.data[idx]['label']

def collate_fn(batch):
    """
    Handles variable number of S-pairs by padding.
    Returns: (padded_features, labels, mask)
    """
    features, labels = zip(*batch)
    
    # Pad sequences: (B, Max_Pairs, Input_Dim)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([len(x) for x in features])
    max_len = padded_features.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    
    labels = torch.stack(labels)
    return padded_features, labels, mask

def train(dist_name, data_path, model_dir, log_dir,
          batches=32, lr=1e-4, epochs=100, patience=10, 
          hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1,
          model_name="SetTransformer"):
    """
    Trains model and saves best checkpoint.
    """
    
    writer = SummaryWriter(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    try:
        full_dataset = GroebnerDataset(data_path)
        print(f"Data loaded from {data_path}.")
    except Exception as e:
        print(f"Error: could not load data from {data_path} ({e}).")
        return
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False, collate_fn=collate_fn)
    
    sample_feat, _ = full_dataset[0]
    input_dim = sample_feat.shape[1]
    print(f"Input Dim: {input_dim} | Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")
    
    model = None
    if model_name.lower() in ("settransformer", "set_transformer", "st"):
        model = SetTransformer(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
    else:
        print(f"Error: Unknown model name '{model_name}'.")
        return
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    l1_loss = nn.L1Loss()

    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print()
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_mae = 0
        
        for features, labels, mask in train_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            value_pred, _ = model(features, mask)
            
            loss = criterion(value_pred.squeeze(), labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += l1_loss(value_pred.squeeze(), labels).item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        
        with torch.no_grad():
            for features, labels, mask in val_loader:
                features, labels, mask = features.to(device), labels.to(device), mask.to(device)
                value_pred, _ = model(features, mask)
                val_loss += criterion(value_pred.squeeze(), labels).item()
                val_mae += l1_loss(value_pred.squeeze(), labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        # Logging
        duration = time.time() - start_time
        scheduler.step(avg_val_loss)
        
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('MAE/Val', avg_val_mae, epoch)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.2f} | "
              f"Val Loss: {avg_val_loss:.2f} (MAE: {avg_val_mae:.2f}) | "
              f"Time: {duration:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{model_dir}/{dist_name}.pth")
            print(f"    >>> New Checkpoint: Best val-loss is {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"    >>> Stopping Early: No improvement for {patience} epochs.")
                break
                
    writer.close()
    print(f"Model Saved: {model_dir}/{dist_name}.pth")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print("Usage: python pretrain.py <DISTRIBUTION>")
        sys.exit()
        
    DIST_NAME = sys.argv[1]
    DATA_PATH = f"../data/tensors/{DIST_NAME}.pt"
    
    MODEL_DIR = f"../model/pretrained/{DIST_NAME}"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    LOG_DIR = f"{MODEL_DIR}/log/"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    train(DIST_NAME, DATA_PATH, MODEL_DIR, LOG_DIR,
          batches=32, lr=1e-4, epochs=100, patience=10,
          hidden_dim=128, num_layers=3, num_heads=4, dropout=0.1)