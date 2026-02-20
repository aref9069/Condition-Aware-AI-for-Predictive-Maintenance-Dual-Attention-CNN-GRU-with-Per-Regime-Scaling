"""
Main pipeline for Dual Attention RUL prediction.
Orchestrates data loading, model training, evaluation, and visualization.

Author: Aref Aasi
"""
import os
import sys
import logging
import numpy as np
import pandas as pd

# Import project modules
from config import (
    SUB_DATASET, WINDOW_SIZE, N_COND_CLUSTERS, USE_DELTAS, VAL_SPLIT,
    HIDDEN_GRU, P_DROP, BATCH_SIZE, EPOCHS, BASE_LR, WEIGHT_DECAY,
    USE_ASYM_MIX, ASYM_WEIGHT, MAX_GRAD_NORM, ES_PATIENCE,
    RESIDUAL_BINS, TOP_N_ERRORS, BASE_DIR, LOG_DIR
)
from utils import set_seed, setup_directories, setup_logging
from data_loader import load_cmapss_data
from model import DualAttentionModel
from training import train_model, evaluate
from losses import r2_score_np
from visualization import (
    plot_learning_curves, plot_parity, plot_residual_hist,
    plot_abs_err_cdf, plot_per_unit_abs_error_topN, plot_residual_vs_true
)


def main():
    """Main pipeline execution."""
    try:
        # Initialize
        set_seed()
        setup_directories()
        setup_logging()
        
        # Parse command line arguments
        data_dir = os.path.join(BASE_DIR, "CMAPSSData")
        debug = False
        if len(sys.argv) > 1:
            data_dir = sys.argv[1].replace('/', '\\')
        if len(sys.argv) > 2 and sys.argv[2].lower() == 'true':
            debug = True
            logging.getLogger().setLevel(logging.DEBUG)
            print("Debug mode enabled")

        # Load and preprocess data
        print("Loading and preprocessing data...")
        X_train, y_train, X_test, y_test, n_features, window_size, test_unit_ids = load_cmapss_data(
            sub_dataset=SUB_DATASET,
            window_size=WINDOW_SIZE,
            data_dir=data_dir,
            n_condition_clusters=N_COND_CLUSTERS,
            add_deltas=USE_DELTAS
        )

        # Time-aware validation split: last 15% windows as validation
        n = len(X_train)
        val_size = max(1, int(VAL_SPLIT * n))
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_trn, y_trn = X_train[:-val_size], y_train[:-val_size]

        # Initialize model
        print("Initializing model...")
        model = DualAttentionModel(
            num_features=n_features,
            seq_len=window_size,
            hidden_gru=HIDDEN_GRU,
            p_drop=P_DROP
        )

        # Train model
        print("Training model...")
        trained_model, history = train_model(
            model, X_trn, y_trn, X_val, y_val,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            base_lr=BASE_LR,
            weight_decay=WEIGHT_DECAY,
            use_asym_mix=USE_ASYM_MIX,
            asym_weight=ASYM_WEIGHT,
            max_grad_norm=MAX_GRAD_NORM,
            es_patience=ES_PATIENCE
        )

        # Plot learning curves
        plot_learning_curves(history["train_loss"], history["val_loss"], history["lr"])

        # Evaluate model
        print("Evaluating model (last-window per unit)...")
        preds, rmse, score = evaluate(trained_model, X_test, y_test)

        # Calculate additional metrics
        err = preds - y_test
        abs_err = np.abs(err)
        mae = float(np.mean(abs_err))
        r2 = float(r2_score_np(y_test, preds))

        # Save predictions
        pred_df = pd.DataFrame({
            "unit": test_unit_ids,
            "true_rul": y_test,
            "pred_rul": preds,
            "error": err,
            "abs_error": abs_err
        })
        pred_csv = os.path.join(LOG_DIR, "predictions.csv")
        pred_df.to_csv(pred_csv, index=False)
        logging.info(f"Saved predictions: {pred_csv}")

        # Save metrics summary
        with open(os.path.join(LOG_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"R2: {r2:.4f}\n")
            f.write(f"PHM Score: {score:.4f}\n")
            f.write(f"Median |error|: {np.median(abs_err):.4f}\n")
            f.write(f"90th pct |error|: {np.percentile(abs_err, 90):.4f}\n")
            f.write(f"95th pct |error|: {np.percentile(abs_err, 95):.4f}\n")

        # Generate all figures
        print("Generating figures...")
        plot_parity(y_test, preds)
        plot_residual_hist(err, bins=RESIDUAL_BINS)
        plot_abs_err_cdf(abs_err)
        plot_per_unit_abs_error_topN(test_unit_ids, abs_err, topN=TOP_N_ERRORS)
        plot_residual_vs_true(y_test, err)

        print("Pipeline complete. Figures and CSV saved in the output folder.")
        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
