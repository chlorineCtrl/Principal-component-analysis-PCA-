"""
Step 3 — class balance and raw feature variance, before any statistics.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from load_data import load_har

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

# Fixed hue order, validated for all-pairs colourblind separation (worst CVD
# dE 6.9, normal-vision 15.6). Grouped so the three dynamic activities read as
# one family and the three static ones as another -- that split is what PC1
# recovers later. Shared by every figure in this project so a colour means the
# same activity everywhere.
ACTIVITY_COLORS = {
    1: "#2a78d6",  # WALKING            blue
    2: "#1baf7a",  # WALKING_UPSTAIRS   aqua
    3: "#008300",  # WALKING_DOWNSTAIRS green
    4: "#eda100",  # SITTING            yellow
    5: "#e34948",  # STANDING           red
    6: "#4a3aa7",  # LAYING             violet
}

# Secondary encoding, required because the palette sits in the CVD warn band.
# Arrows point the way the subject is moving.
ACTIVITY_MARKERS = {1: "o", 2: "^", 3: "v", 4: "s", 5: "D", 6: "X"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd7"


def style_axes(ax, grid_axis="y"):
    """Recessive grid and spines so the data reads first."""
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)


def class_counts(y, label_map):
    """Counts per activity, in activity-id order."""
    return {label_map[i]: int((y == i).sum()) for i in sorted(label_map)}


def feature_variances(X):
    """Per-feature variance of the raw (already min-max scaled) features."""
    return X.var(axis=0, ddof=1)


def plot_eda(y_train, X_train, label_map, out_path):
    counts = class_counts(y_train, label_map)
    variances = feature_variances(X_train)
    log_variances = np.log10(variances)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    names = list(counts)
    values = [counts[n] for n in names]
    colors = [ACTIVITY_COLORS[i] for i in sorted(label_map)]

    bars = ax_left.bar(names, values, color=colors, width=0.68)
    # Direct labels are mandatory relief: three of these hues fall below 3:1
    # contrast on white, so identity must not rest on colour alone.
    for bar, value in zip(bars, values):
        ax_left.text(bar.get_x() + bar.get_width() / 2, value + 18, f"{value:,}",
                     ha="center", va="bottom", fontsize=9, color=INK)

    ax_left.set_title("Training-set class balance", color=INK, fontsize=12, pad=12)
    ax_left.set_ylabel("windows", color=INK_MUTED, fontsize=10)
    ax_left.set_ylim(0, max(values) * 1.12)
    ax_left.set_xticks(range(len(names)))
    ax_left.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8.5)
    style_axes(ax_left)

    ax_right.hist(log_variances, bins=45, color="#2a78d6", edgecolor="white",
                  linewidth=0.5)
    ax_right.set_title(r"Raw feature variance, $\log_{10}$ scale", color=INK,
                       fontsize=12, pad=12)
    ax_right.set_xlabel(r"$\log_{10}(\mathrm{variance})$", color=INK_MUTED,
                        fontsize=10)
    ax_right.set_ylabel("features", color=INK_MUTED, fontsize=10)
    style_axes(ax_right)

    span = log_variances.max() - log_variances.min()
    ax_right.annotate(
        f"spans {span:.1f} orders of magnitude",
        xy=(0.03, 0.94), xycoords="axes fraction",
        fontsize=9, color=INK_MUTED, va="top",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)

    return counts, variances


def run(out_dir=FIGURES_DIR):
    X_train, _, y_train, _, _, _, _, label_map = load_har()
    out_path = Path(out_dir) / "01_eda.png"
    counts, variances = plot_eda(y_train, X_train, label_map, out_path)

    print("class counts (train):")
    for name, value in counts.items():
        print(f"  {name:<20} {value:>5}")
    ratio = max(counts.values()) / min(counts.values())
    print(f"  max/min ratio {ratio:.2f}")

    log_variances = np.log10(variances)
    print("\nraw feature variance:")
    print(f"  min  {variances.min():.3e}")
    print(f"  max  {variances.max():.3e}")
    print(f"  ratio max/min {variances.max() / variances.min():.3e}")
    print(f"  log10 range [{log_variances.min():.2f}, {log_variances.max():.2f}]"
          f"  -> {log_variances.max() - log_variances.min():.1f} orders of magnitude")

    print(f"\nsaved {out_path}")
    return counts, variances


if __name__ == "__main__":
    run()
