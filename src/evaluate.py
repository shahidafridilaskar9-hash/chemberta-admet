"""
evaluate.py
-----------
Evaluation metrics, result visualisation, and calibration
for ChemBERTa ADMET models.

Author: Shahid Afridi Laskar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    mean_squared_error, r2_score, mean_absolute_error,
)
import warnings
warnings.filterwarnings("ignore")


# ─── Classification metrics ───────────────────────────────────────────────────

def classification_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             threshold: float = 0.5) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true    : binary labels
    y_pred    : predicted probabilities
    threshold : float, decision threshold

    Returns
    -------
    dict with ROC-AUC, PR-AUC, accuracy, sensitivity, specificity
    """
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(np.unique(y_true)) < 2:
        return {"error": "Only one class present in y_true"}

    y_binary = (y_pred >= threshold).astype(int)
    tp = np.sum((y_binary == 1) & (y_true == 1))
    tn = np.sum((y_binary == 0) & (y_true == 0))
    fp = np.sum((y_binary == 1) & (y_true == 0))
    fn = np.sum((y_binary == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / len(y_true)

    return {
        "ROC_AUC":     roc_auc_score(y_true, y_pred),
        "PR_AUC":      average_precision_score(y_true, y_pred),
        "Accuracy":    accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "n_samples":   len(y_true),
    }


# ─── Regression metrics ───────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics.

    Returns
    -------
    dict with RMSE, MAE, R²
    """
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE":      rmse,
        "MAE":       mae,
        "R2":        r2,
        "n_samples": len(y_true),
    }


# ─── Comparison table ─────────────────────────────────────────────────────────

def compare_models(results: dict, metric: str = "ROC_AUC") -> pd.DataFrame:
    """
    Build a comparison DataFrame across models and datasets.

    Parameters
    ----------
    results : dict  {model_name: {dataset_name: metrics_dict}}
    metric  : str   metric to compare

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for model_name, datasets in results.items():
        for dataset_name, metrics in datasets.items():
            rows.append({
                "Model":   model_name,
                "Dataset": dataset_name,
                metric:    metrics.get(metric, np.nan),
            })
    return pd.DataFrame(rows).pivot(index="Dataset", columns="Model", values=metric)


# ─── Plotting utilities ───────────────────────────────────────────────────────

def plot_roc_curves(y_true_dict: dict,
                    y_pred_dict: dict,
                    title: str = "ROC Curves",
                    save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    y_true_dict : {model_name: y_true array}
    y_pred_dict : {model_name: y_pred array}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.50)")

    for name in y_true_dict:
        yt = y_true_dict[name]
        yp = y_pred_dict[name]
        mask = ~np.isnan(yt)
        fpr, tpr, _ = roc_curve(yt[mask], yp[mask])
        auc = roc_auc_score(yt[mask], yp[mask])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_regression_scatter(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             label: str = "Model",
                             xlabel: str = "Measured",
                             ylabel: str = "Predicted",
                             save_path: str = None):
    """Scatter plot of measured vs predicted for regression tasks."""
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, alpha=0.5, s=20, label=f"{label}\nR²={r2:.3f}, RMSE={rmse:.3f}")
    lims = [min(yt.min(), yp.min()) - 0.5, max(yt.max(), yp.max()) + 0.5]
    ax.plot(lims, lims, "r--", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{label}: Measured vs Predicted")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    """Plot train/val loss curves from training history dict."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"],   label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
