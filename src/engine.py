from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from .config import KD_ALPHA, KD_TEMPERATURE
from .models import distillation_loss


def train_one_epoch(model, loader, criterion, optimizer, device,
                    teacher=None, kd_enabled: bool = False):
    model.train()
    # Keep running statistics fixed in frozen teacher BatchNorm layers.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        ):
            module.eval()
    if teacher is not None:
        teacher.eval()
    total_loss = 0.0
    total_seen = 0
    all_true, all_pred = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        if kd_enabled:
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss, _, _ = distillation_loss(
                logits, teacher_logits, labels,
                temperature=KD_TEMPERATURE, alpha=KD_ALPHA
            )
        else:
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
    }


@torch.inference_mode()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    all_true, all_pred = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size
        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
    }


def fit_model(model, train_loader, val_loader, device, epochs: int,
              learning_rate: float, checkpoint_path: str | Path,
              teacher=None, kd_enabled: bool = False, model_name: str = "model"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience = 4
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            teacher=teacher, kd_enabled=kd_enabled
        )
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
            f"train loss {row['train_loss']:.4f} | "
            f"val acc {row['val_accuracy']:.4f} | "
            f"val macro-F1 {row['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"[{model_name}] Early stopping after {epoch} epochs.")
            break

    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    return model, pd.DataFrame(history)
