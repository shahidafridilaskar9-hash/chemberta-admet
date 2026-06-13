"""
models.py
---------
ChemBERTa fine-tuning model with single-task and multi-task heads.
Supports both classification and regression endpoints.

Author: Shahid Afridi Laskar
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional


CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"


# ─── Tokenizer helper ────────────────────────────────────────────────────────

def get_tokenizer(model_name: str = CHEMBERTA_MODEL):
    """Load ChemBERTa tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_smiles(smiles_list: list[str],
                    tokenizer,
                    max_length: int = 128) -> dict:
    """
    Tokenize a list of SMILES strings for ChemBERTa.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    tokenizer   : HuggingFace tokenizer
    max_length  : int, maximum token length

    Returns
    -------
    dict with input_ids, attention_mask tensors
    """
    return tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ─── Single-task model ────────────────────────────────────────────────────────

class ChemBERTaSingleTask(nn.Module):
    """
    ChemBERTa encoder with a single task-specific head.

    Architecture:
        ChemBERTa [CLS] token → Dropout → Linear → output

    Supports:
        task_type = 'classification' (binary, sigmoid output)
        task_type = 'regression'     (linear output)
    """

    def __init__(self,
                 model_name: str = CHEMBERTA_MODEL,
                 task_type: str = "classification",
                 dropout: float = 0.1,
                 freeze_encoder: bool = False):
        super().__init__()
        self.task_type = task_type
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)

        if self.task_type == "classification":
            return torch.sigmoid(logits)
        return logits  # regression: raw output


# ─── Multi-task model ─────────────────────────────────────────────────────────

class ChemBERTaMultiTask(nn.Module):
    """
    ChemBERTa encoder with multiple task-specific heads.
    Allows a mix of classification and regression tasks.

    Parameters
    ----------
    task_configs : list of dicts, each with keys:
        'name'      : str
        'task_type' : 'classification' | 'regression'
        'n_classes' : int (1 for binary/regression, >1 for multiclass)
    """

    def __init__(self,
                 model_name: str = CHEMBERTA_MODEL,
                 task_configs: Optional[list] = None,
                 dropout: float = 0.1,
                 freeze_encoder: bool = False):
        super().__init__()

        if task_configs is None:
            # Default: BBBP (classification) + ESOL (regression)
            task_configs = [
                {"name": "bbbp",  "task_type": "classification", "n_classes": 1},
                {"name": "esol",  "task_type": "regression",     "n_classes": 1},
            ]

        self.task_configs = task_configs
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # One linear head per task
        self.heads = nn.ModuleDict({
            cfg["name"]: nn.Linear(hidden_size, cfg["n_classes"])
            for cfg in task_configs
        })

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        results = {}
        for cfg in self.task_configs:
            name = cfg["name"]
            logits = self.heads[name](cls_output).squeeze(-1)
            if cfg["task_type"] == "classification":
                results[name] = torch.sigmoid(logits)
            else:
                results[name] = logits
        return results


# ─── Loss functions ───────────────────────────────────────────────────────────

def get_loss_fn(task_type: str):
    """Return appropriate loss function for task type."""
    if task_type == "classification":
        return nn.BCELoss()
    elif task_type == "regression":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def multitask_loss(predictions: dict,
                   targets: dict,
                   task_configs: list,
                   weights: Optional[dict] = None) -> torch.Tensor:
    """
    Compute weighted sum of per-task losses.

    Parameters
    ----------
    predictions  : dict {task_name: tensor}
    targets      : dict {task_name: tensor}
    task_configs : list of task config dicts
    weights      : optional dict {task_name: float} for loss weighting

    Returns
    -------
    total_loss : scalar tensor
    """
    total = torch.tensor(0.0, requires_grad=True)
    for cfg in task_configs:
        name = cfg["name"]
        loss_fn = get_loss_fn(cfg["task_type"])

        # Handle NaN labels (common in multi-task datasets like Tox21)
        pred = predictions[name]
        tgt = targets[name].float()
        mask = ~torch.isnan(tgt)

        if mask.sum() == 0:
            continue

        task_loss = loss_fn(pred[mask], tgt[mask])
        w = weights.get(name, 1.0) if weights else 1.0
        total = total + w * task_loss

    return total
