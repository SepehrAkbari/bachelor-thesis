import sys

import numpy as np

from torch.utils.data import DataLoader

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dataset import GroebnerDataset, pad_collate
from features import extract_feats


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: linear_regression.py <dataset> <data_type>")
        sys.exit()
    DATASET_NAME = sys.argv[1].lower()
    data_type = sys.argv[2].lower()
    
    if data_type not in ['binomial', 'toric']:
        print("data_type must be either 'binomial' or 'toric'")
        sys.exit()
    N_VARS = 8 if data_type == 'toric' else 3
    
    ds_train = GroebnerDataset(DATASET_NAME, split='train')
    ds_val   = GroebnerDataset(DATASET_NAME, split='val')
    ds_test  = GroebnerDataset(DATASET_NAME, split='test')
    
    loader_train = DataLoader(ds_train, batch_size=1024, collate_fn=pad_collate)
    loader_val = DataLoader(ds_val, batch_size=1024, collate_fn=pad_collate)
    loader_test = DataLoader(ds_test, batch_size=1024, collate_fn=pad_collate)
    
    X_train_raw, y_train = extract_feats(loader_train, n_vars=N_VARS)
    X_val_raw, y_val = extract_feats(loader_val, n_vars=N_VARS)
    X_test, y_test = extract_feats(loader_test, n_vars=N_VARS)
    
    X_train = np.vstack([X_train_raw, X_val_raw])
    y_train = np.concatenate([y_train, y_val])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"Test R2 Score: {test_r2:.3f}")
    print(f"Coefficients: {model.coef_}")
