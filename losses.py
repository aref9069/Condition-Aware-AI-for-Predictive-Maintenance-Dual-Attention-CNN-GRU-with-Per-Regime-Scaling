"""
Loss functions and metrics for RUL prediction.
Includes:
- CMAPSS scoring function
- Asymmetric PHM loss
- R² score


Author: Aref Aasi
"""
import numpy as np
import torch


def cmapss_score(y_true, y_pred):
    """
    CMAPSS asymmetric scoring function.
    
    Penalizes late predictions more heavily than early predictions.
    
    Args:
        y_true: True RUL values (numpy array)
        y_pred: Predicted RUL values (numpy array)
    
    Returns:
        Total score (lower is better)
    """
    diff = y_pred - y_true
    return np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1))


def phm_asym_loss(pred, target):
    """
    Asymmetric loss function for PHM (Prognostics and Health Management).
    
    Penalizes overestimation more than underestimation of RUL.
    
    Args:
        pred: Predicted RUL values (torch tensor)
        target: True RUL values (torch tensor)
    
    Returns:
        Mean asymmetric loss
    """
    diff = pred - target
    over = torch.exp(diff / 10.0) - 1.0
    under = torch.exp(-diff / 13.0) - 1.0
    return torch.mean(torch.where(diff >= 0, over, under))


def r2_score_np(y_true, y_pred):
    """
    Calculate R² (coefficient of determination) score.
    
    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
    
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)
