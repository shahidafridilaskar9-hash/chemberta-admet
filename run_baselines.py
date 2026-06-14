"""
run_baselines.py
----------------
Week 1 deliverable: ECFP4 fingerprint baselines for ADMET prediction.

Trains Random Forest and XGBoost on ECFP4 fingerprints for:
  - BBBP   (classification)  → ROC-AUC, PR-AUC
  - ESOL   (regression)      → RMSE, R2
  - Lipophilicity (regression) → RMSE, R2
  - ClinTox (classification) → ROC-AUC, PR-AUC

Uses scaffold splitting for realistic generalisation estimates.
Results are saved to results/baseline_results.json and printed as a table.

Run:  python run_baselines.py

Author: Shahid Afridi Laskar
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Silence RDKit's verbose deprecation/parse warnings
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.append(str(Path(__file__).resolve().parent))
from src.download_data import download_dataset, get_dataset_config
from src.data_utils import smiles_list_to_ecfp4
from src.splitters import scaffold_split
from src.evaluate import classification_metrics, regression_metrics

warnings.filterwarnings("ignore")

RESULTS = {}


def run_classification(dataset_name: str, label_col: str):
    """Train RF + XGBoost classifiers and return metrics."""
    print(f"\n{'='*60}\n{dataset_name.upper()} — Classification ({label_col})\n{'='*60}")

    df = download_dataset(dataset_name)
    cfg = get_dataset_config(dataset_name)
    smiles_col = cfg["smiles_col"]

    # Drop rows with missing labels
    df = df.dropna(subset=[label_col, smiles_col]).reset_index(drop=True)
    smiles = df[smiles_col].tolist()

    # Scaffold split
    train_idx, valid_idx, test_idx = scaffold_split(smiles)
    print(f"Split: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

    # ECFP4 features
    X_all, valid_fp_idx = smiles_list_to_ecfp4(smiles)
    # Map original indices to fingerprint matrix positions
    idx_map = {orig: pos for pos, orig in enumerate(valid_fp_idx)}

    def subset(indices):
        positions = [idx_map[i] for i in indices if i in idx_map]
        labels = df[label_col].values[[i for i in indices if i in idx_map]]
        return X_all[positions], labels.astype(int)

    X_train, y_train = subset(train_idx)
    X_test, y_test = subset(test_idx)
    print(f"Features: train={X_train.shape}, test={X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_m = classification_metrics(y_test.astype(float), rf_probs)

    # XGBoost
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05,
                        eval_metric="logloss", random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_m = classification_metrics(y_test.astype(float), xgb_probs)

    print(f"RF      -> ROC-AUC: {rf_m['ROC_AUC']:.4f}  PR-AUC: {rf_m['PR_AUC']:.4f}")
    print(f"XGBoost -> ROC-AUC: {xgb_m['ROC_AUC']:.4f}  PR-AUC: {xgb_m['PR_AUC']:.4f}")

    RESULTS[f"{dataset_name}_{label_col}"] = {
        "task": "classification",
        "n_test": int(len(y_test)),
        "RF": {"ROC_AUC": round(rf_m["ROC_AUC"], 4), "PR_AUC": round(rf_m["PR_AUC"], 4)},
        "XGBoost": {"ROC_AUC": round(xgb_m["ROC_AUC"], 4), "PR_AUC": round(xgb_m["PR_AUC"], 4)},
    }


def run_regression(dataset_name: str, label_col: str):
    """Train RF + XGBoost regressors and return metrics."""
    print(f"\n{'='*60}\n{dataset_name.upper()} — Regression ({label_col})\n{'='*60}")

    df = download_dataset(dataset_name)
    cfg = get_dataset_config(dataset_name)
    smiles_col = cfg["smiles_col"]

    df = df.dropna(subset=[label_col, smiles_col]).reset_index(drop=True)
    smiles = df[smiles_col].tolist()

    train_idx, valid_idx, test_idx = scaffold_split(smiles)
    print(f"Split: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

    X_all, valid_fp_idx = smiles_list_to_ecfp4(smiles)
    idx_map = {orig: pos for pos, orig in enumerate(valid_fp_idx)}

    def subset(indices):
        positions = [idx_map[i] for i in indices if i in idx_map]
        labels = df[label_col].values[[i for i in indices if i in idx_map]]
        return X_all[positions], labels.astype(float)

    X_train, y_train = subset(train_idx)
    X_test, y_test = subset(test_idx)
    print(f"Features: train={X_train.shape}, test={X_test.shape}")

    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_m = regression_metrics(y_test, rf_pred)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_m = regression_metrics(y_test, xgb_pred)

    print(f"RF      -> RMSE: {rf_m['RMSE']:.4f}  R2: {rf_m['R2']:.4f}")
    print(f"XGBoost -> RMSE: {xgb_m['RMSE']:.4f}  R2: {xgb_m['R2']:.4f}")

    RESULTS[f"{dataset_name}_{label_col}"] = {
        "task": "regression",
        "n_test": int(len(y_test)),
        "RF": {"RMSE": round(rf_m["RMSE"], 4), "R2": round(rf_m["R2"], 4)},
        "XGBoost": {"RMSE": round(xgb_m["RMSE"], 4), "R2": round(xgb_m["R2"], 4)},
    }


if __name__ == "__main__":
    run_classification("bbbp", "p_np")
    run_regression("esol", "measured log solubility in mols per litre")
    run_regression("lipophilicity", "exp")
    run_classification("clintox", "CT_TOX")

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)

    print(f"\n{'='*60}\nAll baseline results saved to {out_path}\n{'='*60}")
    print(json.dumps(RESULTS, indent=2))
