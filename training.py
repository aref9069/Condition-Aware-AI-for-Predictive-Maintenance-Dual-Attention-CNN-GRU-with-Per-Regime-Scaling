"""
Training and evaluation functions for RUL prediction model.
Includes:
- Training loop with early stopping
- Model evaluation
- Checkpoint saving


Author: Aref Aasi
"""
import os
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from losses import phm_asym_loss, cmapss_score
from config import LOG_DIR, MODEL_DIR


def train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    batch_size=384,
    epochs=120,
    base_lr=8e-4,
    weight_decay=2e-3,
    use_asym_mix=True,
    asym_weight=0.2,
    max_grad_norm=1.0,
    es_patience=12
):
    """
    Train the RUL prediction model.
    
    Args:
        model: PyTorch model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        base_lr: Initial learning rate
        weight_decay: L2 regularization weight
        use_asym_mix: Whether to use asymmetric loss
        asym_weight: Weight for asymmetric loss component
        max_grad_norm: Maximum gradient norm for clipping
        es_patience: Early stopping patience
    
    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    main_crit = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6
    )

    best_state = deepcopy(model.state_dict())
    best_val = float('inf')
    patience = es_patience

    hist_train, hist_val, hist_lr = [], [], []

    for epoch in range(1, epochs+1):
        # Training phase
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = main_crit(pred, yb)
            if use_asym_mix:
                loss = (1.0 - asym_weight) * loss + asym_weight * phm_asym_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            tr_loss += loss.item()

        # Validation phase
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = main_crit(pred, yb)
                if use_asym_mix:
                    loss = (1.0 - asym_weight) * loss + asym_weight * phm_asym_loss(pred, yb)
                va_loss += loss.item()

        tr_loss /= max(1, len(train_loader))
        va_loss /= max(1, len(val_loader))
        scheduler.step(va_loss)

        # Record history
        hist_train.append(tr_loss)
        hist_val.append(va_loss)
        hist_lr.append(optimizer.param_groups[0]['lr'])

        logging.info(
            f"Epoch {epoch:03d}: train={tr_loss:.4f}, val={va_loss:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        print(f"Epoch {epoch:03d}: Train={tr_loss:.4f} | Val={va_loss:.4f}")

        # Early stopping check
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = deepcopy(model.state_dict())
            patience = es_patience
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Save model
    model_path = os.path.join(MODEL_DIR, 'Your model.pth')  # <-- Set this to your desired model 
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")
    print(f"Model saved to {model_path}")

    # Save training history
    hist_df = pd.DataFrame({
        "epoch": np.arange(1, len(hist_train)+1),
        "train_loss": hist_train,
        "val_loss": hist_val,
        "lr": hist_lr
    })
    hist_csv = os.path.join(LOG_DIR, "training_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    logging.info(f"Saved training history: {hist_csv}")

    return model, {"train_loss": hist_train, "val_loss": hist_val, "lr": hist_lr}


def evaluate(model, X_test, y_test, device=None):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels
        device: Device to run evaluation on
    
    Returns:
        preds: Model predictions
        rmse: Root mean squared error
        score: CMAPSS score
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy().ravel()
    
    rmse = float(np.sqrt(np.mean((preds - y_test)**2)))
    score = float(cmapss_score(y_test, preds))
    
    logging.info(f'RMSE: {rmse:.4f}, Score: {score:.4f}')
    print(f'RMSE: {rmse:.4f}, Score: {score:.4f}')
    
    return preds, rmse, score
