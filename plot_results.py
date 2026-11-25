from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ARTIFACT_DIR


def main():
    path = ARTIFACT_DIR / "comparison.csv"
    if not path.exists():
        raise FileNotFoundError("Run python train.py first; comparison.csv is missing.")
    frame = pd.read_csv(path)
    labels = [name.replace("_", "\n") for name in frame["model"]]
    metrics = ["accuracy", "macro_f1"]
    figure, axis = plt.subplots(figsize=(9, 5))
    x = range(len(frame))
    width = 0.35
    axis.bar([i - width / 2 for i in x], frame[metrics[0]], width, label="Accuracy")
    axis.bar([i + width / 2 for i in x], frame[metrics[1]], width, label="Macro F1")
    axis.set_xticks(list(x), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Held-out test performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(ARTIFACT_DIR / "performance_comparison.png", dpi=180)
    print(f"Saved {ARTIFACT_DIR / 'performance_comparison.png'}")


if __name__ == "__main__":
    main()
