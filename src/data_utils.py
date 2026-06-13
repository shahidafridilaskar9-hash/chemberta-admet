"""
data_utils.py
-------------
Dataset loading, SMILES validation, and train/val/test splitting
for MoleculeNet ADMET datasets via DeepChem.

Author: Shahid Afridi Laskar
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings("ignore")


# ─── Dataset configurations ──────────────────────────────────────────────────

DATASET_CONFIG = {
    "bbbp": {
        "loader": "dc.molnet.load_bbbp",
        "task_type": "classification",
        "tasks": ["p_np"],
        "description": "Blood-Brain Barrier Permeability",
    },
    "esol": {
        "loader": "dc.molnet.load_delaney",
        "task_type": "regression",
        "tasks": ["measured log solubility in mols per litre"],
        "description": "Aqueous Solubility (ESOL)",
    },
    "clintox": {
        "loader": "dc.molnet.load_clintox",
        "task_type": "classification",
        "tasks": ["FDA_APPROVED", "CT_TOX"],
        "description": "Clinical Toxicity (ClinTox)",
    },
    "lipophilicity": {
        "loader": "dc.molnet.load_lipo",
        "task_type": "regression",
        "tasks": ["exp"],
        "description": "Lipophilicity (logD)",
    },
    "tox21": {
        "loader": "dc.molnet.load_tox21",
        "task_type": "classification",
        "tasks": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
            "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ],
        "description": "Tox21 Multi-task Toxicity (12 endpoints)",
    },
}


# ─── SMILES utilities ─────────────────────────────────────────────────────────

def is_valid_smiles(smiles: str) -> bool:
    """Return True if SMILES can be parsed by RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> str | None:
    """Return canonical SMILES string, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def filter_valid_smiles(smiles_list: list[str]) -> list[str]:
    """Filter list to valid, canonical SMILES only."""
    valid = []
    for s in smiles_list:
        canon = canonicalize_smiles(s)
        if canon is not None:
            valid.append(canon)
    return valid


# ─── ECFP4 fingerprints (baseline) ───────────────────────────────────────────

def smiles_to_ecfp4(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray | None:
    """
    Convert SMILES to ECFP4 Morgan fingerprint.

    Parameters
    ----------
    smiles : str
    radius : int  Morgan radius (2 = ECFP4)
    nbits  : int  Fingerprint length

    Returns
    -------
    np.ndarray of shape (nbits,) or None if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp)


def smiles_list_to_ecfp4(smiles_list: list[str],
                          radius: int = 2,
                          nbits: int = 2048) -> tuple[np.ndarray, list[int]]:
    """
    Batch convert SMILES list to ECFP4 matrix.

    Returns
    -------
    X      : np.ndarray of shape (n_valid, nbits)
    valid_idx : list of indices where conversion succeeded
    """
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        fp = smiles_to_ecfp4(smi, radius=radius, nbits=nbits)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    return np.array(fps), valid_idx


# ─── RDKit descriptor set (for interpretability) ─────────────────────────────

def smiles_to_rdkit_descriptors(smiles: str) -> dict | None:
    """Compute a small set of physicochemical descriptors from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def lipinski_filter(smiles: str) -> dict:
    """
    Apply Lipinski Rule of Five.
    Returns dict with individual rule checks and overall pass/fail.
    """
    desc = smiles_to_rdkit_descriptors(smiles)
    if desc is None:
        return {"valid": False}
    return {
        "valid": True,
        "MW_ok": desc["MolWt"] <= 500,
        "LogP_ok": desc["LogP"] <= 5,
        "HBD_ok": desc["NumHDonors"] <= 5,
        "HBA_ok": desc["NumHAcceptors"] <= 10,
        "Ro5_pass": (
            desc["MolWt"] <= 500 and
            desc["LogP"] <= 5 and
            desc["NumHDonors"] <= 5 and
            desc["NumHAcceptors"] <= 10
        ),
        **desc,
    }


# ─── DeepChem dataset loader ──────────────────────────────────────────────────

def load_moleculenet_dataset(dataset_name: str,
                              splitter: str = "scaffold",
                              reload: bool = True):
    """
    Load a MoleculeNet dataset via DeepChem.

    Parameters
    ----------
    dataset_name : str   One of keys in DATASET_CONFIG
    splitter     : str   'scaffold' | 'random' | 'stratified'
    reload       : bool  Cache dataset locally

    Returns
    -------
    tasks, (train, valid, test), transformers
    """
    import deepchem as dc

    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_CONFIG.keys())}"
        )

    featurizer = dc.feat.DummyFeaturizer()  # raw SMILES — tokenized later
    splitter_obj = dc.splits.ScaffoldSplitter() if splitter == "scaffold" else \
                   dc.splits.RandomSplitter()

    loader_map = {
        "bbbp":          dc.molnet.load_bbbp,
        "esol":          dc.molnet.load_delaney,
        "clintox":       dc.molnet.load_clintox,
        "lipophilicity": dc.molnet.load_lipo,
        "tox21":         dc.molnet.load_tox21,
    }

    loader = loader_map[dataset_name]
    tasks, datasets, transformers = loader(
        featurizer=featurizer,
        splitter=splitter,
        reload=reload,
    )
    return tasks, datasets, transformers


def dataset_to_dataframe(dc_dataset, tasks: list[str]) -> pd.DataFrame:
    """
    Convert a DeepChem Dataset to a pandas DataFrame with SMILES + labels.
    """
    smiles = [mol for mol in dc_dataset.ids]
    labels = dc_dataset.y

    df = pd.DataFrame({"smiles": smiles})
    for i, task in enumerate(tasks):
        df[task] = labels[:, i] if labels.ndim > 1 else labels
    return df
