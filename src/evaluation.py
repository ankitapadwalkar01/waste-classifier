
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .config import CLASS_NAMES
from .models import count_parameters


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()

    y_true, y_pred = [], []

    for images, labels in loader:
        logits = model(images.to(device))

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    return y_true, y_pred


def measure_latency(
    model,
    device,
    image_size: int,
    warmup: int = 10,
    repetitions: int = 80
) -> dict[str, float]:

    model.eval()

    tensor = torch.randn(
        1, 3, image_size, image_size, device=device
    )

    with torch.inference_mode():

        for _ in range(warmup):
            _ = model(tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []

        for _ in range(repetitions):
            start = time.perf_counter()

            _ = model(tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            timings.append(
                (time.perf_counter() - start) * 1000.0
            )

    median_ms = float(np.median(timings))

    return {
        "latency_ms_median": median_ms,
        "latency_ms_mean": float(np.mean(timings)),
        "fps_batch1": 1000.0 / max(median_ms, 1e-9),
    }


def checkpoint_size_mb(model) -> float:
    """
    Calculate serialized model size in memory.
    Avoids Windows temporary-file locking errors.
    """

    buffer = io.BytesIO()

    torch.save(model.state_dict(), buffer)

    size_bytes = buffer.getbuffer().nbytes

    return size_bytes / (1024 * 1024)


def evaluate_model(
    name: str,
    model,
    test_loader,
    device,
    image_size: int,
    output_dir: str | Path
) -> dict:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()

    y_true, y_pred = predict_loader(
        model, test_loader, device
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    # Generate confusion matrix
    figure, axis = plt.subplots(figsize=(10, 8))

    image = axis.imshow(cm, cmap="Blues")

    axis.figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {name}",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=40,
        ha="right",
        rotation_mode="anchor"
    )

    threshold = cm.max() / 2 if cm.size else 0

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):

            axis.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row, col] > threshold
                    else "black"
                ),
                fontsize=8
            )

    figure.tight_layout()

    figure.savefig(
        output_dir / f"confusion_{name}.png",
        dpi=180
    )

    plt.close(figure)

    # Save classification report
    pd.DataFrame(report).T.to_csv(
        output_dir / f"classification_report_{name}.csv"
    )

    # Measure inference latency
    latency = measure_latency(
        model,
        device,
        image_size
    )

    if device.type == "cpu":

        latency["latency_ms_median_cpu"] = (
            latency["latency_ms_median"]
        )

        latency["fps_batch1_cpu"] = (
            latency["fps_batch1"]
        )

    else:

        cpu_device = torch.device("cpu")

        cpu_model = model.to(cpu_device).eval()

        cpu_latency = measure_latency(
            cpu_model,
            cpu_device,
            image_size,
            warmup=5,
            repetitions=40
        )

        latency["latency_ms_median_cpu"] = (
            cpu_latency["latency_ms_median"]
        )

        latency["fps_batch1_cpu"] = (
            cpu_latency["fps_batch1"]
        )

    # Collect model metrics
    result = {
        "model": name,

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),

        "parameters_total": int(
            count_parameters(model)
        ),

        "parameters_trainable": int(
            count_parameters(
                model,
                trainable_only=True
            )
        ),

        "state_dict_size_mb": float(
            checkpoint_size_mb(model)
        ),

        **latency,
    }

    # Save individual model metrics
    with open(
        output_dir / f"metrics_{name}.json",
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(result, handle, indent=2)

    return result


def save_comparison(
    results: list[dict],
    output_dir: str | Path
) -> pd.DataFrame:

    frame = pd.DataFrame(results)

    preferred = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "weighted_f1",
        "parameters_total",
        "state_dict_size_mb",
        "latency_ms_median",
        "fps_batch1",
        "latency_ms_median_cpu",
        "fps_batch1_cpu",
    ]

    frame = frame[
        [
            column
            for column in preferred
            if column in frame.columns
        ]
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(
        output_dir / "comparison.csv",
        index=False
    )

    return frame
