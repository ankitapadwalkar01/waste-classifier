from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch

from src.config import (
    ARTIFACT_DIR, CLASS_NAMES, IMAGE_SIZE, KD_ALPHA, KD_TEMPERATURE,
    SEED, SPLIT_FILE,
)
from src.data import create_splits, find_class_directory, make_loaders, set_seed
from src.engine import fit_model
from src.evaluation import evaluate_model, save_comparison
from src.models import LightweightWasteCNN, build_teacher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet teacher and a lightweight CNN with/without knowledge distillation."
    )
    parser.add_argument("--data-dir", default="data/raw/extracted",
                        help="Directory containing the extracted RealWaste dataset.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--teacher-epochs", type=int, default=6)
    parser.add_argument("--student-epochs", type=int, default=12)
    parser.add_argument("--teacher-lr", type=float, default=1e-4)
    parser.add_argument("--student-lr", type=float, default=8e-4)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet weights for the teacher.")
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes ({len(CLASS_NAMES)}): {', '.join(CLASS_NAMES)}")
    print(f"Distillation: temperature={KD_TEMPERATURE}, alpha={KD_ALPHA}")

    class_root = find_class_directory(args.data_dir)
    create_splits(class_root, SPLIT_FILE, force=args.rebuild_splits, seed=args.seed)
    loaders = make_loaders(
        class_root, SPLIT_FILE, image_size=args.image_size,
        batch_size=args.batch_size, num_workers=args.workers, seed=args.seed
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train the powerful teacher.
    print("\n=== EXPERIMENT 1: RESNET-50 TEACHER ===")
    teacher = build_teacher(len(CLASS_NAMES), pretrained=not args.no_pretrained)
    teacher, teacher_history = fit_model(
        teacher, loaders["train"], loaders["validation"], device,
        epochs=args.teacher_epochs, learning_rate=args.teacher_lr,
        checkpoint_path=ARTIFACT_DIR / "teacher_resnet50_best.pth",
        model_name="Teacher ResNet-50",
    )
    teacher_history.to_csv(ARTIFACT_DIR / "history_teacher.csv", index=False)

    # Establish one common student initialisation for a fair CE vs KD comparison.
    set_seed(args.seed + 100)
    initial_student = LightweightWasteCNN(num_classes=len(CLASS_NAMES))
    initial_state = copy.deepcopy(initial_student.state_dict())
    # Recreate loaders from the same seed so CE and KD see the same sampling schedule.
    student_loaders = make_loaders(
        class_root, SPLIT_FILE, image_size=args.image_size,
        batch_size=args.batch_size, num_workers=args.workers, seed=args.seed
    )

    # 2. Student trained with ground-truth labels only.
    print("\n=== EXPERIMENT 2: STUDENT WITHOUT DISTILLATION ===")
    student_ce = LightweightWasteCNN(num_classes=len(CLASS_NAMES))
    student_ce.load_state_dict(initial_state)
    set_seed(args.seed + 200)
    student_ce, ce_history = fit_model(
        student_ce, student_loaders["train"], student_loaders["validation"], device,
        epochs=args.student_epochs, learning_rate=args.student_lr,
        checkpoint_path=ARTIFACT_DIR / "student_ce_best.pth",
        model_name="Student CE",
    )
    ce_history.to_csv(ARTIFACT_DIR / "history_student_ce.csv", index=False)

    # 3. Identically-initialised student trained with hard labels + teacher soft targets.
    print("\n=== EXPERIMENT 3: STUDENT WITH KNOWLEDGE DISTILLATION ===")
    student_kd = LightweightWasteCNN(num_classes=len(CLASS_NAMES))
    student_kd.load_state_dict(initial_state)
    set_seed(args.seed + 200)
    student_loaders = make_loaders(
        class_root, SPLIT_FILE, image_size=args.image_size,
        batch_size=args.batch_size, num_workers=args.workers, seed=args.seed
    )
    student_kd, kd_history = fit_model(
        student_kd, student_loaders["train"], student_loaders["validation"], device,
        epochs=args.student_epochs, learning_rate=args.student_lr,
        checkpoint_path=ARTIFACT_DIR / "student_kd_best.pth",
        teacher=teacher, kd_enabled=True,
        model_name="Student KD",
    )
    kd_history.to_csv(ARTIFACT_DIR / "history_student_kd.csv", index=False)

    # Save self-describing metadata for the demo and future reproducibility.
    for filename, architecture in [
        ("teacher_resnet50_best.pth", "resnet50"),
        ("student_ce_best.pth", "lightweight_waste_cnn"),
        ("student_kd_best.pth", "lightweight_waste_cnn"),
    ]:
        path = ARTIFACT_DIR / filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "architecture": architecture,
            "class_names": CLASS_NAMES,
            "image_size": args.image_size,
            "seed": args.seed,
        })
        torch.save(payload, path)

    # 4. Evaluate every model exactly once on the untouched test partition.
    print("\n=== FINAL HELD-OUT TEST EVALUATION ===")
    results = []
    for name, model in [
        ("teacher_resnet50", teacher),
        ("student_ce", student_ce),
        ("student_kd", student_kd),
    ]:
        print(f"Evaluating {name}...")
        results.append(evaluate_model(
            name, model, loaders["test"], device, args.image_size, ARTIFACT_DIR
        ))
    comparison = save_comparison(results, ARTIFACT_DIR)
    print("\n=== COMPARISON ===")
    print(comparison.to_string(index=False))

    final_split_frame = pd.read_csv(SPLIT_FILE)
    summary = {
        "dataset": "UCI RealWaste",
        "class_names": CLASS_NAMES,
        "device_used_for_training": str(device),
        "image_size": args.image_size,
        "seed": args.seed,
        "split_counts": {
            split: int((final_split_frame["split"] == split).sum())
            for split in ["train", "validation", "test"]
        },
        "distillation": {"temperature": KD_TEMPERATURE, "alpha_ce": KD_ALPHA},
        "results": results,
        "interpretation_note": (
            "Use the held-out test results as measured outcomes. Do not claim KD improved "
            "accuracy unless student_kd exceeds student_ce on the reported metric."
        ),
    }
    with open(ARTIFACT_DIR / "experiment_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTraining and evaluation completed.")
    print(f"Artifacts saved in: {ARTIFACT_DIR.resolve()}")
    print("Run the demo with: streamlit run app.py")


if __name__ == "__main__":
    main()
