"""
predict.py
----------
Single-compound and batch inference for trained ChemBERTa ADMET models.

Author: Shahid Afridi Laskar
"""

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from src.data_utils import canonicalize_smiles, lipinski_filter
from src.models import get_tokenizer, tokenize_smiles


def predict_single(smiles: str,
                   model,
                   tokenizer,
                   task_type: str = "classification",
                   device: str = "cpu") -> dict:
    """
    Predict ADMET property for a single SMILES string.

    Parameters
    ----------
    smiles    : str   input SMILES
    model     : trained ChemBERTaSingleTask
    tokenizer : ChemBERTa tokenizer
    task_type : 'classification' | 'regression'
    device    : str

    Returns
    -------
    dict with prediction score, Lipinski filter, canonical SMILES
    """
    canon = canonicalize_smiles(smiles)
    if canon is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    encoding = tokenize_smiles([canon], tokenizer)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    model.eval()
    with torch.no_grad():
        pred = model(encoding["input_ids"], encoding["attention_mask"])
        score = pred.item()

    result = {
        "smiles": canon,
        "score": score,
        "task_type": task_type,
    }

    if task_type == "classification":
        result["prediction"] = "Active" if score >= 0.5 else "Inactive"
        result["confidence"] = max(score, 1 - score)

    result["lipinski"] = lipinski_filter(canon)
    return result


def predict_batch(smiles_list: list[str],
                  model,
                  tokenizer,
                  task_type: str = "classification",
                  batch_size: int = 32,
                  device: str = "cpu") -> pd.DataFrame:
    """
    Predict ADMET properties for a list of SMILES.

    Returns
    -------
    pd.DataFrame with smiles, score, and Lipinski descriptors
    """
    model.eval()
    all_scores = []
    valid_smiles = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        canonical = [canonicalize_smiles(s) for s in batch]
        valid = [(s, c) for s, c in zip(batch, canonical) if c is not None]

        if not valid:
            continue

        orig, canon = zip(*valid)
        encoding = tokenize_smiles(list(canon), tokenizer)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            preds = model(encoding["input_ids"], encoding["attention_mask"])
            scores = preds.cpu().numpy()

        for smi, score in zip(canon, scores):
            all_scores.append({"smiles": smi, "score": float(score)})
            valid_smiles.append(smi)

    df = pd.DataFrame(all_scores)

    # Add Lipinski descriptors
    lipinski_rows = [lipinski_filter(s) for s in df["smiles"]]
    lipinski_df = pd.DataFrame(lipinski_rows)

    return pd.concat([df.reset_index(drop=True),
                      lipinski_df.reset_index(drop=True)], axis=1)
