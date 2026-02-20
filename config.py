"""
Configuration settings for Dual Attention RUL prediction.

Author: Aref Aasi
"""
import os

# =========================
# Reproducibility
# =========================
SEED = 42

# =========================
# Paths
# =========================
BASE_DIR = r"Your path here"  # <-- Set this to your desired base directory
LOG_DIR = os.path.join(BASE_DIR, "logs")
FIG_DIR = os.path.join(LOG_DIR, "figures")
MODEL_DIR = BASE_DIR  # save model in main folder
FIG_DATA_XLSX = os.path.join(LOG_DIR, "figure_data.xlsx")

# =========================
# Data Parameters
# =========================
SUB_DATASET = 'FD002'
WINDOW_SIZE = 80
N_COND_CLUSTERS = 8
USE_DELTAS = True
VAL_SPLIT = 0.15

# =========================
# Model Hyperparameters
# =========================
HIDDEN_GRU = 256
P_DROP = 0.2

# =========================
# Training Hyperparameters
# =========================
BATCH_SIZE = 384
EPOCHS = 120
BASE_LR = 8e-4
WEIGHT_DECAY = 2e-3
USE_ASYM_MIX = True
ASYM_WEIGHT = 0.2
MAX_GRAD_NORM = 1.0
ES_PATIENCE = 12

# =========================
# Plotting Parameters
# =========================
RESIDUAL_BINS = 30
TOP_N_ERRORS = 30
