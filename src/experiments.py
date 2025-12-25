import torch
import pandas as pd
from src.models import BuchbergerTransformer
from src.data_loader import get_data_loader
from src.train import train_one_epoch, log_cosh_loss

def run_cross_distribution_test(train_name, test_names, data_path, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Set Transformer
    # Removed num_generators argument to fix TypeError
    model = BuchbergerTransformer(input_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train_loader = get_data_loader(data_path, train_name)
    
    print(f"Training on {train_name}...")
    for epoch in range(epochs):
        # The model's forward() now handles padding internally
        loss = train_one_epoch(model, train_loader, optimizer, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
            
    # 2. Cross-Evaluation
    results = {}
    model.eval()
    for t_name in test_names:
        # Note: shuffle=False for reproducible testing
        test_loader = get_data_loader(data_path, t_name, shuffle=False)
        preds, actuals = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                # Removed manual padding here; Model handles (..., 6) -> (..., 8) automatically
                preds.append(model(x).cpu())
                actuals.append(y)
        
        y_p, y_a = torch.cat(preds), torch.cat(actuals)
        
        # Calculate R-squared
        ss_res = torch.sum((y_a - y_p)**2)
        ss_tot = torch.sum((y_a - y_a.mean())**2)
        r2 = 1 - (ss_res / ss_tot)
        results[t_name] = r2.item()
        
    return results