"""
visualize_results.py

Generates evaluation visualizations from model outputs and saves them to reports/figures/.

Outputs:
    confusion_matrix_val.png
    confusion_matrix_test.png
    precision_recall_f1.png
    feature_coefficients.png
    dataset_composition.png

Usage:
    python model/visualize_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

MODEL_DIR = Path("model")
DATA_DIR  = Path("data/processed")
OUT_DIR   = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

FRAUD_COLOR = "#D85A30"
REAL_COLOR  = "#1D9E75"
NEUTRAL     = "#888780"

with open(MODEL_DIR / "evaluation_report.json") as f:
    report = json.load(f)

with open(MODEL_DIR / "top_features.json") as f:
    features = json.load(f)

with open(DATA_DIR / "dataset_metadata.json") as f:
    metadata = json.load(f)


def plot_confusion_matrix(cm, split_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_arr = np.array(cm)

    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted real", "Predicted fraud"], fontsize=11)
    ax.set_yticklabels(["Actual real", "Actual fraud"], fontsize=11)

    for i in range(2):
        for j in range(2):
            color = "white" if cm_arr[i, j] > cm_arr.max() / 2 else "#2C2C2A"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_title(f"Confusion matrix - {split_name}", fontsize=13, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = OUT_DIR / f"confusion_matrix_{split_name.lower()}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


plot_confusion_matrix(report["val"]["confusion_matrix"],  "val")
plot_confusion_matrix(report["test"]["confusion_matrix"], "test")


def plot_prf():
    metrics = ["precision", "recall", "f1-score"]
    splits  = ["val", "test"]
    classes = ["real", "fraud"]
    colors  = [REAL_COLOR, FRAUD_COLOR]

    x       = np.arange(len(metrics))
    width   = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(9, 5))

    for ci, cls in enumerate(classes):
        for si, split in enumerate(splits):
            idx   = ci * 2 + si
            vals  = [report[split]["classification_report"][cls][m] for m in metrics]
            alpha = 1.0 if si == 1 else 0.55
            ax.bar(x + offsets[idx] * width, vals, width,
                   color=colors[ci], alpha=alpha, label=f"{cls} ({split})")

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-score"], fontsize=12)
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision, recall, and F1 by class (val vs test)", fontsize=13, pad=12)

    legend_elements = [
        mpatches.Patch(color=REAL_COLOR,  alpha=0.55, label="real - val"),
        mpatches.Patch(color=REAL_COLOR,  alpha=1.0,  label="real - test"),
        mpatches.Patch(color=FRAUD_COLOR, alpha=0.55, label="fraud - val"),
        mpatches.Patch(color=FRAUD_COLOR, alpha=1.0,  label="fraud - test"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, framealpha=0.4)

    plt.tight_layout()
    path = OUT_DIR / "precision_recall_f1.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


plot_prf()


def plot_features():
    fraud_feats = features["top_fraud_signals"][:10]
    real_feats  = features["top_real_signals"][:10]

    all_feats = fraud_feats + real_feats
    names     = [f["feature"].replace("indian_", "* ") for f in all_feats]
    coefs     = [f["coefficient"] for f in all_feats]
    colors    = [FRAUD_COLOR if c > 0 else REAL_COLOR for c in coefs]

    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(names))
    ax.barh(y, coefs, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color=NEUTRAL, linewidth=0.8)
    ax.set_xlabel("Logistic regression coefficient", fontsize=11)
    ax.set_title("Top fraud and real signals (* = Indian hand-crafted feature)", fontsize=12, pad=12)
    ax.invert_yaxis()

    fraud_patch = mpatches.Patch(color=FRAUD_COLOR, alpha=0.85, label="-> fraud")
    real_patch  = mpatches.Patch(color=REAL_COLOR,  alpha=0.85, label="-> real")
    ax.legend(handles=[fraud_patch, real_patch], fontsize=10, framealpha=0.4)

    plt.tight_layout()
    path = OUT_DIR / "feature_coefficients.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


plot_features()


def plot_dataset():
    sources = metadata["source_distribution"]
    labels  = list(sources.keys())
    values  = list(sources.values())
    colors  = [FRAUD_COLOR, REAL_COLOR, "#378ADD", "#BA7517"][:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    wedges, texts, autotexts = axes[0].pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.75, textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[0].set_title("Dataset sources", fontsize=12, pad=10)

    split_names  = ["train", "val", "test"]
    split_fraud  = []
    split_real   = []
    for s in split_names:
        df = pd.read_csv(f"data/processed/{s}.csv")
        split_fraud.append((df["label"] == 1).sum())
        split_real.append((df["label"] == 0).sum())

    x     = np.arange(len(split_names))
    width = 0.35
    axes[1].bar(x - width/2, split_real,  width, label="real",  color=REAL_COLOR,  alpha=0.85)
    axes[1].bar(x + width/2, split_fraud, width, label="fraud", color=FRAUD_COLOR, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Train", "Val", "Test"], fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Class distribution per split", fontsize=12, pad=10)
    axes[1].legend(fontsize=10, framealpha=0.4)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("Fraudulent Job Detection - dataset overview", fontsize=13, y=1.01)
    plt.tight_layout()
    path = OUT_DIR / "dataset_composition.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


plot_dataset()

print(f"\nAll figures saved to {OUT_DIR}/")