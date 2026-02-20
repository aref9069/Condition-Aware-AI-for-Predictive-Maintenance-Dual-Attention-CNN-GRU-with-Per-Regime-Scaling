"""
Visualization functions for RUL prediction results.
Includes:
- Learning curves
- Parity plots
- Residual analysis
- Error distributions


Author: Aref Aasi
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from config import LOG_DIR, FIG_DIR, FIG_DATA_XLSX


def _sheet_name(name: str) -> str:
    """Ensure Excel sheet names are within 31 character limit."""
    return name[:31] if len(name) > 31 else name


def save_figure_data(name_base: str, df: pd.DataFrame):
    """
    Save figure data to CSV and Excel.
    
    Args:
        name_base: Base name for the output files
        df: DataFrame containing the figure data
    """
    csv_path = os.path.join(LOG_DIR, f"{name_base}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logging.info(f"Saved figure data CSV: {csv_path}")

    mode = "a" if os.path.exists(FIG_DATA_XLSX) else "w"
    try:
        with pd.ExcelWriter(FIG_DATA_XLSX, engine="openpyxl", mode=mode, 
                           if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=_sheet_name(name_base), index=False)
        logging.info(f"Saved/updated Excel sheet '{name_base}' in {FIG_DATA_XLSX}")
    except Exception as e:
        # Fallback to xlsxwriter if openpyxl not available
        if "openpyxl" in str(e).lower():
            with pd.ExcelWriter(FIG_DATA_XLSX, engine="xlsxwriter", mode="w") as writer:
                df.to_excel(writer, sheet_name=_sheet_name(name_base), index=False)
            logging.info(f"Saved Excel (xlsxwriter fallback) sheet '{name_base}' in {FIG_DATA_XLSX}")
        else:
            logging.error(f"Excel write failed for '{name_base}': {e}")


def _savefig(fig, name_base):
    """Save figure in both PNG and PDF formats."""
    png = os.path.join(FIG_DIR, f"{name_base}.png")
    pdf = os.path.join(FIG_DIR, f"{name_base}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved figures: {png}, {pdf}")


def plot_learning_curves(train_losses, val_losses, lrs=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        lrs: Optional list of learning rates per epoch
    """
    # Save underlying data
    epochs = np.arange(1, len(train_losses) + 1)
    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss": np.asarray(train_losses, dtype=float),
        "val_loss": np.asarray(val_losses, dtype=float),
        "lr": np.asarray(lrs, dtype=float) if lrs is not None else np.nan
    })
    save_figure_data("learning_curves", df)

    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _savefig(fig, "learning_curves")


def plot_parity(y_true, y_pred):
    """
    Plot predicted vs true RUL values (parity plot).
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
    """
    # Save underlying data
    df = pd.DataFrame({
        "true_rul": np.asarray(y_true, dtype=float),
        "pred_rul": np.asarray(y_pred, dtype=float)
    })
    save_figure_data("parity_pred_vs_true", df)

    # Plot
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=18, alpha=0.6, edgecolors="none")
    lim = max(np.max(y_true), np.max(y_pred)) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Predicted vs True")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "parity_pred_vs_true")


def plot_residual_hist(residuals, bins=30):
    """
    Plot histogram of prediction residuals.
    
    Args:
        residuals: Prediction residuals (pred - true)
        bins: Number of histogram bins
    """
    # Compute and save histogram table
    counts, edges = np.histogram(residuals, bins=bins)
    df = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": counts
    })
    save_figure_data("residual_hist", df)

    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.hist(residuals, bins=bins, alpha=0.85)
    ax.set_xlabel("Residual (Pred − True)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Histogram")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "residual_hist")


def plot_abs_err_cdf(abs_err):
    """
    Plot cumulative distribution function of absolute errors.
    
    Args:
        abs_err: Absolute error values
    """
    # Compute and save ECDF
    n = len(abs_err)
    x = np.sort(abs_err)
    y = np.arange(1, n+1) / n
    p50 = np.percentile(abs_err, 50)
    p90 = np.percentile(abs_err, 90)
    
    df = pd.DataFrame({"abs_error_sorted": x, "ecdf": y})
    save_figure_data("abs_error_cdf", df)
    
    # Store percentiles separately
    meta = pd.DataFrame({"stat": ["P50", "P90"], "value": [p50, p90]})
    save_figure_data("abs_error_cdf_meta", meta)

    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=2)
    ax.axvline(p50, color="gray", linestyle="--")
    ax.text(p50, 0.52, f"P50={p50:.1f}", rotation=90, va="bottom")
    ax.axvline(p90, color="gray", linestyle="--")
    ax.text(p90, 0.12, f"P90={p90:.1f}", rotation=90, va="bottom")
    ax.set_xlabel("|Error|")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Absolute Error CDF")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "abs_error_cdf")


def plot_per_unit_abs_error_topN(unit_ids, abs_err, topN=30):
    """
    Plot top N units by absolute error.
    
    Args:
        unit_ids: Unit identifiers
        abs_err: Absolute error values
        topN: Number of top errors to display
    """
    # Compute and save ranking table
    order = np.argsort(-abs_err)
    uu = np.array(unit_ids)[order][:topN]
    ee = np.asarray(abs_err, dtype=float)[order][:topN]
    df = pd.DataFrame({"unit_id": uu, "abs_error": ee})
    save_figure_data("per_unit_abs_error_top30", df)

    # Plot
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(uu)), ee, width=0.8)
    ax.set_xticks(range(len(uu)))
    ax.set_xticklabels([str(u) for u in uu], rotation=90)
    ax.set_xlabel("Unit ID (Top by |error|)")
    ax.set_ylabel("|Error|")
    ax.set_title(f"Top {topN} Absolute Errors")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, "per_unit_abs_error_top30")


def plot_residual_vs_true(y_true, residuals):
    """
    Plot residuals vs true RUL values to check for bias.
    
    Args:
        y_true: True RUL values
        residuals: Prediction residuals
    """
    # Save underlying data
    df = pd.DataFrame({
        "true_rul": np.asarray(y_true, dtype=float),
        "residual": np.asarray(residuals, dtype=float)
    })
    save_figure_data("residual_vs_true", df)

    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, residuals, s=18, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Residual (Pred − True)")
    ax.set_title("Residual vs True")
    ax.grid(True, alpha=0.3)
    _savefig(fig, "residual_vs_true")
