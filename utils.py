"""
Utility functions for reproducibility and common operations.


Author: Aref Aasi
"""
import os
import random
import logging
import numpy as np
import torch
from config import SEED, LOG_DIR, FIG_DIR, MODEL_DIR


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_directories():
    """Create necessary directories if they don't exist."""
    for d in [LOG_DIR, FIG_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)


def setup_logging():
    """Configure logging system."""
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "pipeline.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print(f"Logs: {os.path.join(LOG_DIR, 'pipeline.log')}")
    print(f"Figures: {FIG_DIR}")
