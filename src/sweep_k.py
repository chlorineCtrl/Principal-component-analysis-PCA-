"""
what k=102 actually costs: SVM-RBF test accuracy and fit time across
the whole range of component counts.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from load_data import load_har
from preprocess import standardize
from eda import INK, INK_MUTED, style_axes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

SERIES_1 = "#2a78d6"
SERIES_2 = "#e34948"
ACCENT = "#4a3aa7"

K_VALUES = (2, 5, 10, 20, 35, 50, 75, 102, 150, 200, 300, 561)
CHOSEN_K = 102
RANDOM_STATE = 42


def sweep(Z_train, Z_test, y_train, y_test, k_values=K_VALUES):
    """One full PCA fit, then slice it — the first k components of the full fit
    are exactly what PCA(n_components=k) would return."""
    pca = PCA(n_components=max(k_values), random_state=RANDOM_STATE).fit(Z_train)
    P_train = pca.transform(Z_train)
    P_test = pca.transform(Z_test)
    cum = np.cumsum(pca.explained_variance_ratio_)

    rows = []
    for k in k_values:
        model = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)

        start = time.perf_counter()
        model.fit(P_train[:, :k], y_train)
        fit_time = time.perf_counter() - start

        accuracy = accuracy_score(y_test, model.predict(P_test[:, :k]))
        rows.append({
            "k": k,
            "variance_retained_pct": float(cum[k - 1] * 100),
            "test_accuracy": float(accuracy),
            "fit_time_s": fit_time,
        })
        print(f"  k={k:>3}  acc={accuracy:.4f}  var={cum[k - 1] * 100:5.1f}%  "
              f"fit={fit_time:.2f}s")

    return pd.DataFrame(rows)


def plot_sweep(frame, out_path):
    """Two stacked panels on a shared x-axis rather than one chart with two
    y-scales: with two scales the crossing point is an artefact of where the
    axes happen to be anchored."""
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9.5, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")

    best = frame.loc[frame["test_accuracy"].idxmax()]
    chosen = frame.loc[frame["k"] == CHOSEN_K].iloc[0]

    ax_top.plot(frame["k"], frame["test_accuracy"] * 100, color=SERIES_1,
                linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=0.8)
    ax_top.plot([best["k"]], [best["test_accuracy"] * 100], marker="o",
                markersize=9, color=SERIES_2, markeredgecolor="white",
                markeredgewidth=1.4, zorder=3)
    
    ax_top.annotate(f"best  k={int(best['k'])}   {best['test_accuracy'] * 100:.2f}%",
                    xy=(best["k"], best["test_accuracy"] * 100),
                    xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=9.5, color=SERIES_2)
    ax_top.annotate(f"k={CHOSEN_K}   {chosen['test_accuracy'] * 100:.2f}%",
                    xy=(CHOSEN_K, chosen["test_accuracy"] * 100),
                    xytext=(8, -20), textcoords="offset points",
                    fontsize=9.5, color=INK)
    ax_top.set_ylim(frame["test_accuracy"].min() * 100 - 3, 99)
    ax_top.set_ylabel("test accuracy (%)", color=INK_MUTED, fontsize=10)
    ax_top.set_title("Accuracy saturates long before the last component",
                     color=INK, fontsize=12, pad=12)
    style_axes(ax_top, grid_axis="both")

    ax_bottom.plot(frame["k"], frame["fit_time_s"], color=ACCENT, linewidth=2,
                   marker="o", markersize=5, markeredgecolor="white",
                   markeredgewidth=0.8)
    ax_bottom.set_ylabel("SVM fit time (s)", color=INK_MUTED, fontsize=10)
    ax_bottom.set_xlabel("components kept (log scale)", color=INK_MUTED, fontsize=10)
    ax_bottom.set_xscale("log")
    ax_bottom.set_xticks(list(K_VALUES))
    ax_bottom.set_xticklabels([str(k) for k in K_VALUES], fontsize=8.5)
    ax_bottom.minorticks_off()
    style_axes(ax_bottom, grid_axis="both")

    for ax in (ax_top, ax_bottom):
        ax.axvline(CHOSEN_K, color=INK_MUTED, linewidth=1, linestyle=":")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def run(out_dir=FIGURES_DIR, results_dir=RESULTS_DIR):
    out_dir = Path(out_dir)
    results_dir = Path(results_dir)

    X_train, X_test, y_train, y_test, *_ = load_har()
    Z_train, Z_test, _ = standardize(X_train, X_test)

    print("sweeping k:")
    frame = sweep(Z_train, Z_test, y_train, y_test)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "accuracy_vs_k.csv"
    frame.to_csv(out_csv, index=False)
    print(f"  saved {out_csv.name}")

    plot_sweep(frame, out_dir / "09_accuracy_vs_k.png")

    best = frame.loc[frame["test_accuracy"].idxmax()]
    chosen = frame.loc[frame["k"] == CHOSEN_K].iloc[0]
    gap = (best["test_accuracy"] - chosen["test_accuracy"]) * 100
    print(f"\nbest k={int(best['k'])} at {best['test_accuracy']:.4f}; "
          f"k={CHOSEN_K} gives {chosen['test_accuracy']:.4f} ({gap:.2f}pp lower) "
          f"using {CHOSEN_K / best['k']:.2f}x the dimensions")

    return frame


if __name__ == "__main__":
    run()
