"""
train.py
--------
Training loop for ChemBERTa ADMET models with early stopping,
learning rate scheduling, and checkpoint saving.

Author: Shahid Afridi Laskar
"""

import os
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class SMILESDataset(Dataset):
    """
    Simple dataset wrapping tokenized SMILES + labels.

    Parameters
    ----------
    encodings : dict from tokenizer (input_ids, attention_mask)
    labels    : np.ndarray of shape (n_samples,) or (n_samples, n_tasks)
    """

    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


# ─── Training loop ────────────────────────────────────────────────────────────

def train_single_task(model,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       task_type: str = "classification",
                       epochs: int = 10,
                       lr: float = 2e-5,
                       warmup_ratio: float = 0.1,
                       patience: int = 3,
                       save_path: str = "results/best_model.pt",
                       device: Optional[str] = None):
    """
    Fine-tune a ChemBERTaSingleTask model.

    Parameters
    ----------
    model       : ChemBERTaSingleTask
    train_loader: DataLoader
    val_loader  : DataLoader
    task_type   : 'classification' | 'regression'
    epochs      : int
    lr          : float  learning rate
    warmup_ratio: float  fraction of steps for LR warmup
    patience    : int    early stopping patience
    save_path   : str    path to save best checkpoint
    device      : str    'cuda' | 'cpu' | None (auto-detect)

    Returns
    -------
    history : dict with train_loss and val_loss per epoch
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.BCELoss() if task_type == "classification" else torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # ── Validation ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        # ── Early stopping ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return history
