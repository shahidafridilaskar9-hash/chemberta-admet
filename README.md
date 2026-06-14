# ChemBERTa-ADMET: Multi-Task ADMET Property Prediction via Fine-Tuned Molecular Transformers

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)

> **Author:** Shahid Afridi Laskar  
> **GitHub:** [shahidafridilaskar9-hash](https://github.com/shahidafridilaskar9-hash)  
> **Related publication:** [GLUT3 inhibitor CADD study, TCABSE-J 2024](https://www.tcabse.org/_files/ugd/a1924b_8a833c05c29a4a0692aaec8663bcb172.pdf)

---

## Research Question

ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) filtering is the single largest attrition bottleneck in early drug discovery — over 40% of clinical failures are attributable to poor pharmacokinetic or toxicity profiles. Existing ADMET predictors rely predominantly on hand-crafted molecular fingerprints (ECFP, MACCS) or shallow ML models that lack the ability to capture long-range structural context in molecules.

**This project asks:** Can a transformer pretrained on molecular SMILES strings (ChemBERTa) be fine-tuned jointly across multiple ADMET endpoints to yield a single, interpretable, deployment-ready predictor that outperforms fingerprint baselines?

---

## Why This Matters for CADD Pipelines

In my published GLUT3 inhibitor study (TCABSE-J, 2024), ADMET filtering was applied post-hoc using SwissADME — a single-point, non-learnable tool. This project systematises and improves that step:

- Replaces static rule-based filters with a learned, data-driven model
- Produces calibrated probability scores rather than binary pass/fail
- Is directly embeddable into virtual screening pipelines (AutoDock Vina → ChemBERTa-ADMET → ranked candidates)
- Connects to the TeachOpenCADD open-source CADD framework philosophy

---

## Endpoints Modelled

| Endpoint | Dataset | Task | Biological Relevance |
|---|---|---|---|
| Blood-Brain Barrier Permeability | BBBP (MoleculeNet) | Binary classification | CNS drug access |
| Aqueous Solubility (logS) | ESOL (MoleculeNet) | Regression | Oral bioavailability |
| Clinical Toxicity | ClinTox (MoleculeNet) | Binary classification | Safety filtering |
| Multitask Toxicity | Tox21 (MoleculeNet) | 12-task classification | Regulatory toxicology |
| Lipophilicity (logD) | Lipophilicity (MoleculeNet) | Regression | Membrane permeability |

---

## Approach

```
SMILES strings
      ↓
ChemBERTa tokenizer (SMILES-aware BPE)
      ↓
ChemBERTa encoder (pretrained on 77M ZINC SMILES)
      ↓
[CLS] token embedding → task-specific heads
      ↓
Multi-task fine-tuning (classification + regression)
      ↓
Calibrated ADMET scores per compound
```

**Baseline comparison:** ECFP4 fingerprints + Random Forest / XGBoost on same datasets.

**Evaluation:** ROC-AUC (classification), RMSE + R² (regression), per-task and averaged.

---

## Project Structure

```
chemberta-admet/
├── README.md
├── requirements.txt
├── environment.yml
├── notebooks/
│   ├── 01_data_exploration.ipynb       # MoleculeNet dataset loading + EDA
│   ├── 02_fingerprint_baseline.ipynb   # ECFP4 + RF/XGBoost baselines
│   ├── 03_chemberta_finetuning.ipynb   # ChemBERTa fine-tuning per endpoint
│   ├── 04_multitask_model.ipynb        # Multi-task head architecture
│   └── 05_results_analysis.ipynb       # Benchmark comparison + figures
├── src/
│   ├── __init__.py
│   ├── download_data.py    # Direct MoleculeNet download (no DeepChem)
│   ├── splitters.py        # Bemis-Murcko scaffold splitting
│   ├── data_utils.py       # SMILES cleaning, ECFP4, Lipinski filter
│   ├── models.py           # ChemBERTa model + task heads
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # Metrics, calibration, visualisation
│   └── predict.py          # Single-compound inference
├── run_baselines.py        # Week 1 deliverable: ECFP4 baselines
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── figures/
    └── .gitkeep
```

---

## Current Status

- [x] Repository structure and documentation
- [x] Data loading pipeline (direct MoleculeNet download, no DeepChem dependency)
- [x] Scaffold splitting (Bemis-Murcko) for realistic generalisation tests
- [x] ECFP4 fingerprint baselines (BBBP, ESOL, Lipophilicity, ClinTox)
- [ ] ChemBERTa single-task fine-tuning (next: run on GPU)
- [ ] Conformal prediction layer for uncertainty-aware screening
- [ ] Multi-task head architecture
- [ ] HuggingFace Spaces deployment

## Baseline Results (ECFP4 + scaffold split)

Reproducible via `python run_baselines.py`. Scaffold splitting makes these
deliberately challenging (test set contains unseen scaffolds).

| Dataset | Task | Model | ROC-AUC | PR-AUC | RMSE | R² |
|---|---|---|---|---|---|---|
| BBBP | Classification | RF | 0.690 | 0.710 | – | – |
| BBBP | Classification | XGBoost | 0.657 | 0.682 | – | – |
| ClinTox | Classification | RF | 0.678 | 0.337 | – | – |
| ClinTox | Classification | XGBoost | **0.885** | **0.547** | – | – |
| ESOL | Regression | RF | – | – | **1.644** | 0.396 |
| Lipophilicity | Regression | XGBoost | – | – | **0.879** | 0.366 |

These are the performance bars ChemBERTa must beat. The next step is
transformer fine-tuning, followed by a **conformal prediction layer** that
adds statistically guaranteed uncertainty estimates — turning point
predictions into trustworthy prediction sets for compound prioritisation.

---

## Quickstart

```bash
git clone https://github.com/shahidafridilaskar9-hash/chemberta-admet
cd chemberta-admet

# Core dependencies (Python 3.9–3.13)
pip install rdkit scikit-learn pandas numpy matplotlib seaborn requests xgboost

# Run the ECFP4 baselines (downloads data automatically)
python run_baselines.py

# For ChemBERTa fine-tuning (notebook 03), also install:
pip install torch transformers
# GPU recommended — Google Colab free tier works well
```

---

## Relation to TeachOpenCADD

This project is designed in the spirit of [TeachOpenCADD](https://github.com/volkamerlab/teachopencadd) — reproducible, notebook-based, open-source CADD education. The notebooks follow a talktorial structure (theory → implementation → results → discussion) and are accessible to researchers with a biological background.

A natural extension would be a TeachOpenCADD talktorial on transformer-based ADMET prediction — something this project is prototyping independently.

---

## References

1. Chithrananda et al. (2020). ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. *NeurIPS Workshop*.
2. Wu et al. (2018). MoleculeNet: A Benchmark for Molecular Machine Learning. *Chemical Science*.
3. Ramsundar et al. (2019). Deep Learning for the Life Sciences. O'Reilly.
4. Volkamer et al. TeachOpenCADD. [github.com/volkamerlab/teachopencadd](https://github.com/volkamerlab/teachopencadd)
5. Laskar SA (2024). In silico identification and QSAR analysis of GLUT3 inhibitors. *TCABSE-J*, Vol.1, Issue 8, pp.19–24.
