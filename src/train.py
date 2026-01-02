import sys
import copy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import r2_score

from dataset import GroebnerDataset, pad_collate


class EarlyStopping:
    """
    Stops training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = np.inf
        self.best_weights = None
        self.counter = 0
        self.status = ""

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.status = f"New best: {val_loss:.4f}"
            return False
        else:
            self.counter += 1
            self.status = f"No improvement. {self.counter}/{self.patience}"
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
            return False

def train_model(model, loaders, criterion, optimizer, scheduler=None, num_epochs=100, device='cuda', patience=15):
    """
    Universal training loop for regression models
    
    Args:
        model: PyTorch model
        loaders: Dict containing 'train' and 'val' DataLoaders
        criterion: Loss function (e.g., nn.MSELoss)
        optimizer: Optimizer (e.g., Adam)
        scheduler: LR Scheduler (optional)
        num_epochs: Max epochs
        device: 'cuda' or 'cpu'
        patience: Early stopping patience
    """
    model = model.to(device)
    stopper = EarlyStopping(patience=patience)
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        
        ## Training
        
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for inputs, targets, lengths in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, lengths) 
            
            outputs = outputs.squeeze()
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(loaders['train'].dataset)
        history['train_loss'].append(epoch_loss)
        
        ## Validation
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, lengths in loaders['val']:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs, lengths)
                outputs = outputs.squeeze()
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_val_loss = val_loss / len(loaders['val'].dataset)
        val_r2 = r2_score(all_targets, all_preds)
        
        history['val_loss'].append(epoch_val_loss)
        history['val_r2'].append(val_r2)
        
        ## Convergence check
        
        if scheduler:
            scheduler.step(epoch_val_loss)
            
        current_lr = optimizer.param_groups[0]['lr']
        stop = stopper(epoch_val_loss, model)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss {epoch_loss:.2f} | Val Loss {epoch_val_loss:.2f} | Val R2 {val_r2:.3f} | LR {current_lr:.1e} | {stopper.status}")
        
        if stop:
            print("Early stopping triggered.")
            break
            
    return model, history


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: train.py <model> <dataset>")
        sys.exit()
    model_name = sys.argv[1].lower()
    dataset_name = sys.argv[2].lower()

    ds_train = GroebnerDataset(dataset_name, split='train', padding=16)
    ds_val = GroebnerDataset(dataset_name, split='val', padding=16)

    loaders = {
        'train': DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=pad_collate),
        'val': DataLoader(ds_val, batch_size=32, collate_fn=pad_collate)
    }
    
    if model_name == 'gru' or model_name == 'rnn':
        from gru import GRU
        model_name = GRU
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_name(input_dim=16, hidden_dim=128).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_model, history = train_model(
        model, 
        loaders, 
        criterion, 
        optimizer, 
        scheduler=scheduler, 
        device='cuda',
        patience=20
    )

    print(f"Final Validation R2: {history['val_r2'][-1]}")