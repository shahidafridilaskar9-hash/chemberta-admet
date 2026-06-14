"""
download_data.py
----------------
Download MoleculeNet ADMET datasets directly from the public DeepChem
S3 bucket as CSV files. This avoids requiring the full DeepChem package
(which has heavy dependencies and limited Python 3.13 support).

Datasets are cached locally in the data/ folder.

Author: Shahid Afridi Laskar
"""

import os
import gzip
import shutil
import requests
import pandas as pd
from pathlib import Path


# Public DeepChem MoleculeNet dataset URLs
DATASET_URLS = {
    "bbbp": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "filename": "BBBP.csv",
        "smiles_col": "smiles",
        "label_cols": ["p_np"],
        "task_type": "classification",
    },
    "esol": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "filename": "delaney-processed.csv",
        "smiles_col": "smiles",
        "label_cols": ["measured log solubility in mols per litre"],
        "task_type": "regression",
    },
    "lipophilicity": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "filename": "Lipophilicity.csv",
        "smiles_col": "smiles",
        "label_cols": ["exp"],
        "task_type": "regression",
    },
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "filename": "clintox.csv.gz",
        "smiles_col": "smiles",
        "label_cols": ["FDA_APPROVED", "CT_TOX"],
        "task_type": "classification",
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "filename": "tox21.csv.gz",
        "smiles_col": "smiles",
        "label_cols": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ],
        "task_type": "classification",
    },
}


def get_data_dir() -> Path:
    """Return the data directory path, creating it if needed."""
    # data/ is one level up from src/
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_dataset(dataset_name: str, force: bool = False) -> pd.DataFrame:
    """
    Download (if needed) and load a MoleculeNet dataset as a DataFrame.

    Parameters
    ----------
    dataset_name : str   one of DATASET_URLS keys
    force        : bool  re-download even if cached

    Returns
    -------
    pd.DataFrame with SMILES + label columns
    """
    if dataset_name not in DATASET_URLS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_URLS.keys())}"
        )

    config = DATASET_URLS[dataset_name]
    data_dir = get_data_dir()
    raw_path = data_dir / config["filename"]

    # Download if not cached
    if force or not raw_path.exists():
        print(f"Downloading {dataset_name} from {config['url']} ...")
        resp = requests.get(config["url"], timeout=120)
        resp.raise_for_status()
        with open(raw_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved to {raw_path}")
    else:
        print(f"Using cached {dataset_name} at {raw_path}")

    # Load (handle gzip)
    if config["filename"].endswith(".gz"):
        csv_path = raw_path.with_suffix("")  # remove .gz
        if force or not csv_path.exists():
            with gzip.open(raw_path, "rb") as f_in:
                with open(csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(raw_path)

    return df


def get_dataset_config(dataset_name: str) -> dict:
    """Return the configuration dict for a dataset."""
    return DATASET_URLS[dataset_name]


if __name__ == "__main__":
    # Quick test: download all datasets and report shapes
    for name in DATASET_URLS:
        try:
            df = download_dataset(name)
            cfg = DATASET_URLS[name]
            print(f"\n{name.upper()}")
            print(f"  Shape: {df.shape}")
            print(f"  SMILES column: '{cfg['smiles_col']}'")
            print(f"  Label columns: {cfg['label_cols']}")
            print(f"  Columns present: {list(df.columns)[:8]}")
        except Exception as e:
            print(f"\n{name.upper()} — FAILED: {e}")
