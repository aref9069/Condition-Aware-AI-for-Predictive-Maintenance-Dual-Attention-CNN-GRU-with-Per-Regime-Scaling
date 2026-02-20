# Dual Attention RUL Prediction

A modular deep learning framework for Remaining Useful Life (RUL) prediction using dual attention mechanisms on the CMAPSS (C-MAPSS) turbofan engine degradation dataset.

## Project Structure

```
.
├── config.py           # Configuration settings and hyperparameters
├── utils.py            # Utility functions (reproducibility, logging)
├── data_loader.py      # Data loading and preprocessing
├── model.py            # Neural network architecture
├── losses.py           # Custom loss functions and metrics
├── training.py         # Training and evaluation loops
├── visualization.py    # Plotting and visualization functions
├── main.py             # Main pipeline orchestrator
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Module Descriptions

### 1. `config.py`
Central configuration file containing:
- Paths (data, logs, figures, models)
- Dataset parameters (window size, RUL cap, clusters)
- Model hyperparameters (hidden units, dropout)
- Training hyperparameters (learning rate, batch size, epochs)
- Reproducibility seed

### 2. `utils.py`
Utility functions for:
- Setting random seeds for reproducibility
- Creating necessary directories
- Setting up logging system

### 3. `data_loader.py`
Data processing pipeline:
- CMAPSS dataset loading from text files
- Feature engineering (sensor selection, delta features)
- KMeans-based per-condition normalization
- Sliding window sequence generation
- Train/test split with last-window evaluation

### 4. `model.py`
Neural network architecture:
- **ChannelAttention**: Attention mechanism for feature channels
- **ConvBlock**: Convolutional block with residual connections
- **SequenceAttention**: Temporal attention mechanism
- **DualAttentionModel**: Complete model combining:
  - Multi-scale CNN with channel attention
  - Bidirectional GRU for temporal modeling
  - Sequence attention for aggregation
  - Fully connected prediction head

### 5. `losses.py`
Loss functions and metrics:
- CMAPSS asymmetric scoring function
- PHM (Prognostics and Health Management) asymmetric loss
- R² coefficient of determination

### 6. `training.py`
Training infrastructure:
- Training loop with gradient clipping
- Early stopping based on validation loss
- Learning rate scheduling (ReduceLROnPlateau)
- Model checkpoint saving
- Evaluation on test set

### 7. `visualization.py`
Comprehensive visualization suite:
- Learning curves (train/val loss over epochs)
- Parity plots (predicted vs true RUL)
- Residual histograms
- Absolute error CDF
- Per-unit error analysis
- Residual vs true RUL scatter plots
- Automatic data export to CSV and Excel

### 8. `main.py`
Pipeline orchestrator that:
- Initializes all components
- Loads and preprocesses data
- Trains the model
- Evaluates performance
- Generates all visualizations
- Saves results and metrics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python main.py
```

### Custom Data Directory
```bash
python main.py /path/to/CMAPSSData
```

### Debug Mode
```bash
python main.py /path/to/CMAPSSData true
```

## Key Features

### Data Processing
- **Per-condition normalization**: Uses KMeans clustering on operating conditions
- **Delta features**: Optional first-difference features for capturing trends
- **Last-window evaluation**: Test set uses only the final window per unit
- **RUL capping**: Configurable maximum RUL value (default: 125 cycles)

### Model Architecture
- **Dual attention**: Combines channel and sequence attention mechanisms
- **Multi-scale CNN**: Multiple convolutional layers with different kernel sizes and dilations
- **Residual connections**: Skip connections in convolutional blocks
- **Dropout regularization**: Configurable dropout for preventing overfitting

### Training Strategy
- **Asymmetric loss**: Penalizes late predictions more than early ones
- **Mixed loss**: Combination of SmoothL1Loss and asymmetric PHM loss
- **Gradient clipping**: Prevents exploding gradients
- **Early stopping**: Stops training when validation loss plateaus
- **Learning rate scheduling**: Adaptive learning rate reduction

### Output
The pipeline generates:
- **Model checkpoint**: `Your model should be here.pth`  # <-- Set this to your desired model 
- **Training history**: CSV with loss and learning rate per epoch
- **Predictions**: CSV with per-unit predictions and errors
- **Metrics summary**: Text file with RMSE, MAE, R², score, percentiles
- **Figures**: PNG and PDF formats for all visualizations
- **Figure data**: CSV and Excel exports of all plot data

## Configuration

Key parameters can be modified in `config.py`:

```python
# Data parameters
SUB_DATASET = 'FD002'
WINDOW_SIZE = 80
N_COND_CLUSTERS = 8

# Model parameters
HIDDEN_GRU = 256
P_DROP = 0.2

# Training parameters
BATCH_SIZE = 384
EPOCHS = 120
BASE_LR = 8e-4
ES_PATIENCE = 12
```

## Extending the Framework

### Adding New Models
Create a new model class in `model.py` inheriting from `nn.Module`

### Custom Loss Functions
Add new loss functions to `losses.py`

### Additional Visualizations
Add plotting functions to `visualization.py` following the existing pattern

### Different Datasets
Modify `data_loader.py` to support new dataset formats

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

See `requirements.txt` for complete dependencies.

## License

This project is licensed under the MIT License.

## Citation

If you use this framework in your research or industrial applications, please cite the following paper:

Aasi, A. (2026).
Condition-Aware Dual-Attention CNN-GRU Framework for Remaining Useful Life Prediction under Multi-Operating Regimes.
Expert Systems with Applications, 2026, Article 131582.
https://doi.org/10.1016/j.eswa.2026.131582

ScienceDirect link:
https://www.sciencedirect.com/science/article/pii/S0957417426004951