"""
Dual Attention RUL Prediction Package
A modular framework for Remaining Useful Life prediction using dual attention mechanisms.
"""

__version__ = "1.0.0"
__author__ = "Aref Aasi"

# Import main components for easy access
from .model import DualAttentionModel, ChannelAttention, SequenceAttention
from .data_loader import load_cmapss_data
from .training import train_model, evaluate
from .losses import cmapss_score, phm_asym_loss, r2_score_np
from .utils import set_seed, setup_directories, setup_logging

__all__ = [
    'DualAttentionModel',
    'ChannelAttention',
    'SequenceAttention',
    'load_cmapss_data',
    'train_model',
    'evaluate',
    'cmapss_score',
    'phm_asym_loss',
    'r2_score_np',
    'set_seed',
    'setup_directories',
    'setup_logging'
]
