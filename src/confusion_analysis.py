"""
where the SVM-RBF is wrong, before and after PCA.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.svm import SVC

from load_data import load_har
from preprocess import standardize
from eda import GRID, INK, INK_MUTED

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

CHOSEN_K = 102
RANDOM_STATE = 42

# Counts are magnitude
COUNT_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#ffffff", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95",
              "#0d366b"],
)


def fit_svm(X_train, y_train, X_test):
    model = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _panel(ax, matrix, names, title):
    # Colour by row share so classes with different support stay comparable
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(shares, cmap=COUNT_CMAP, vmin=0, vmax=1)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if shares[i, j] > 0.55 else INK)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("predicted", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("true", color=INK_MUTED, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, pad=10)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ax.spines.values():
        side.set_color(GRID)


def plot_confusion(before, after, names, acc_before, acc_after, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")

    _panel(axes[0], before, names,
           f"before PCA, 561 dimensions   (accuracy {acc_before:.4f})")
    _panel(axes[1], after, names,
           f"after PCA, {CHOSEN_K} dimensions   (accuracy {acc_after:.4f})")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def worst_pairs(matrix, names, top=3):
    """Largest off-diagonal confusions, symmetrized over the two directions."""
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((matrix[i, j] + matrix[j, i], names[i], names[j]))
    return sorted(pairs, reverse=True)[:top]


def run(out_dir=FIGURES_DIR, results_dir=RESULTS_DIR):
    out_dir = Path(out_dir)
    results_dir = Path(results_dir)

    X_train, X_test, y_train, y_test, _, _, _, label_map = load_har()
    Z_train, Z_test, _ = standardize(X_train, X_test)

    pca = PCA(n_components=CHOSEN_K, random_state=RANDOM_STATE).fit(Z_train)
    P_train, P_test = pca.transform(Z_train), pca.transform(Z_test)

    print("fitting SVM-RBF before and after PCA:")
    pred_before = fit_svm(Z_train, y_train, Z_test)
    pred_after = fit_svm(P_train, y_train, P_test)

    acc_before = accuracy_score(y_test, pred_before)
    acc_after = accuracy_score(y_test, pred_after)

    labels = sorted(label_map)
    names = [label_map[i] for i in labels]
    cm_before = confusion_matrix(y_test, pred_before, labels=labels)
    cm_after = confusion_matrix(y_test, pred_after, labels=labels)

    plot_confusion(cm_before, cm_after, names, acc_before, acc_after,
                   out_dir / "10_confusion_matrices.png")

    report = classification_report(y_test, pred_after, labels=labels,
                                   target_names=names, digits=4,
                                   output_dict=True)
    frame = pd.DataFrame(report).transpose()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "classification_report.csv"
    frame.to_csv(out_csv)
    print(f"  saved {out_csv.name}")

    print(f"\nclassification report, after PCA (k={CHOSEN_K}):")
    print(classification_report(y_test, pred_after, labels=labels,
                                target_names=names, digits=4))

    for title, matrix in (("before PCA", cm_before), ("after PCA", cm_after)):
        print(f"largest confusions, {title}:")
        for count, a, b in worst_pairs(matrix, names):
            print(f"  {a} <-> {b}: {count}")

    return frame, cm_before, cm_after


if __name__ == "__main__":
    run()
