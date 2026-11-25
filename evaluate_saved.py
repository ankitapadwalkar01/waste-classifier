
from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.config import (
    ARTIFACT_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    SEED,
    SPLIT_FILE,
)
from src.data import find_class_directory, make_loaders
from src.models import LightweightWasteCNN, build_teacher, count_parameters


def load_saved_model(name, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    architecture = checkpoint["architecture"]

    if architecture == "resnet50":
        model = build_teacher(len(CLASS_NAMES), pretrained=False)
    elif architecture == "lightweight_waste_cnn":
        model = LightweightWasteCNN(num_classes=len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"Loaded {name}: {checkpoint_path.name}")
    return model


@torch.inference_mode()
def evaluate(name, model, loader, device, checkpoint_path):
    y_true = []
    y_pred = []

    total_forward_time = 0.0
    total_images = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        start = time.perf_counter()
        outputs = model(images)
        total_forward_time += time.perf_counter() - start

        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        total_images += labels.size(0)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    latency_ms = (
        total_forward_time / total_images * 1000
        if total_images else 0
    )
    fps = (
        total_images / total_forward_time
        if total_forward_time else 0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        ARTIFACT_DIR / f"classification_report_{name}.csv"
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    )
    fig, ax = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(
        ARTIFACT_DIR / f"confusion_{name}.png",
        dpi=180,
    )
    plt.close(fig)

    result = {
        "model": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "parameters": count_parameters(model),
        "checkpoint_size_mb": (
            Path(checkpoint_path).stat().st_size / (1024 * 1024)
        ),
        "inference_latency_ms_per_image": latency_ms,
        "inference_fps": fps,
        "test_images": total_images,
    }

    print(f"\n{name}")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return result


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # Do not generate a new split. Use the original training split.
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Original split file not found: {SPLIT_FILE}. "
            "Evaluation stopped to avoid creating a different test split."
        )

    class_root = find_class_directory("data/raw/extracted")

    loaders = make_loaders(
        class_root,
        SPLIT_FILE,
        image_size=IMAGE_SIZE,
        batch_size=24,
        num_workers=0,
        seed=SEED,
    )

    test_loader = loaders["test"]
    print(f"Test images: {len(test_loader.dataset)}")

    checkpoints = [
        (
            "teacher_resnet50",
            ARTIFACT_DIR / "teacher_resnet50_best.pth",
        ),
        (
            "student_ce",
            ARTIFACT_DIR / "student_ce_best.pth",
        ),
        (
            "student_kd",
            ARTIFACT_DIR / "student_kd_best.pth",
        ),
    ]

    results = []

    for name, path in checkpoints:
        print(f"\nEvaluating {name}...")
        model = load_saved_model(name, path, device)

        result = evaluate(
            name, model, test_loader, device, path
        )
        results.append(result)

        del model

    comparison = pd.DataFrame(results)
    comparison.to_csv(
        ARTIFACT_DIR / "comparison.csv",
        index=False,
    )

    with open(
        ARTIFACT_DIR / "test_evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\n========== FINAL TEST COMPARISON ==========")
    print(comparison.to_string(index=False))
    print("\nEvaluation completed. No models were trained.")
    print(f"Results saved in: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
