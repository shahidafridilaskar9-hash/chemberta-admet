"""
splitters.py
------------
Train/validation/test splitting for molecular datasets.
Implements scaffold splitting (Bemis-Murcko) for realistic, challenging
splits that test generalisation to novel chemical scaffolds — the
standard for ADMET benchmarking.

Author: Shahid Afridi Laskar
"""

import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings("ignore")


def generate_scaffold(smiles: str, include_chirality: bool = False) -> str | None:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    Returns the scaffold SMILES, or None if the molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    return scaffold


def scaffold_split(smiles_list: list[str],
                   frac_train: float = 0.8,
                   frac_valid: float = 0.1,
                   frac_test: float = 0.1,
                   seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """
    Split dataset by Bemis-Murcko scaffold.

    Molecules sharing a scaffold are kept in the same split, so the
    test set contains scaffolds unseen during training. This gives a
    realistic, pessimistic estimate of model generalisation.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    frac_train, frac_valid, frac_test : split fractions (must sum to 1)
    seed : random seed for reproducibility

    Returns
    -------
    train_idx, valid_idx, test_idx : lists of integer indices
    """
    assert abs(frac_train + frac_valid + frac_test - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    # Group molecule indices by scaffold
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = generate_scaffold(smi)
        if scaffold is not None:
            scaffolds[scaffold].append(idx)

    # Sort scaffold sets from largest to smallest (standard practice)
    scaffold_sets = sorted(
        scaffolds.values(),
        key=lambda x: (len(x), x[0]),
        reverse=True,
    )

    n_total = sum(len(s) for s in scaffold_sets)
    n_train = int(np.floor(frac_train * n_total))
    n_valid = int(np.floor(frac_valid * n_total))

    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= n_train:
            train_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= n_valid:
            valid_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    return train_idx, valid_idx, test_idx


def random_split(n_samples: int,
                 frac_train: float = 0.8,
                 frac_valid: float = 0.1,
                 frac_test: float = 0.1,
                 seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """Random train/valid/test split of indices."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_train = int(frac_train * n_samples)
    n_valid = int(frac_valid * n_samples)
    return (
        idx[:n_train].tolist(),
        idx[n_train:n_train + n_valid].tolist(),
        idx[n_train + n_valid:].tolist(),
    )
