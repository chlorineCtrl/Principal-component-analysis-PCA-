"""
the figures for the Visualization section, plus the reconstruction
error table that checks truncation error against the discarded eigenvalues.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from load_data import load_har
from preprocess import standardize
from pca_scratch import MyPCA
from component_selection import all_criteria, k_scree_elbow
from eda import (ACTIVITY_COLORS, ACTIVITY_MARKERS, GRID, INK, INK_MUTED,
                 style_axes)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

SERIES_1 = "#2a78d6"
SERIES_2 = "#e34948"
ACCENT = "#4a3aa7"

# Correlation is polarity, so a diverging ramp: two poles, neutral gray midpoint.
CORR_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f2b2a"],
)

SCREE_WINDOW = 60
HEATMAP_N = 30
BIPLOT_ARROWS = 6
RECON_KS = (2, 5, 10, 20, 50, 102, 150, 200, 300, 400, 500, 561)
CHOSEN_K = 102


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  saved {path.name}")


def _mean_abs_offdiag(C):
    mask = ~np.eye(C.shape[0], dtype=bool)
    return np.abs(C[mask]).mean()


def _reconstruct(Xc_mean, W, scores, k):
    return scores[:, :k] @ W[:k] + Xc_mean


def plot_scree_and_cumulative(eigvals, cum, elbow, out_path):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    idx = np.arange(1, SCREE_WINDOW + 1)
    y = eigvals[:SCREE_WINDOW]

    # Log y: lambda_1 is 8x lambda_2 and 260x lambda_60, so a linear axis would
    # flatten everything after the first point and hide the Kaiser line.
    ax_l.plot(idx, y, color=SERIES_1, linewidth=2, marker="o", markersize=4,
              markeredgecolor="white", markeredgewidth=0.6)
    ax_l.set_yscale("log")
    ax_l.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle="--")
    # Anchored left: by k=55 the curve itself has dropped onto the lambda=1 line.
    ax_l.annotate(r"Kaiser threshold $\lambda=1$", xy=(1, 1.0),
                  xytext=(6, 6), textcoords="offset points", ha="left",
                  fontsize=9, color=INK_MUTED)
    ax_l.axvline(elbow, color=SERIES_2, linewidth=1.2, linestyle=":")
    ax_l.annotate(f"scree elbow, k={elbow}", xy=(elbow, y[elbow - 1]),
                  xytext=(14, 10), textcoords="offset points",
                  fontsize=9, color=SERIES_2)
    ax_l.set_title(f"Scree plot, first {SCREE_WINDOW} components", color=INK,
                   fontsize=12, pad=12)
    ax_l.set_xlabel("component", color=INK_MUTED, fontsize=10)
    ax_l.set_ylabel("eigenvalue (log scale)", color=INK_MUTED, fontsize=10)
    style_axes(ax_l, grid_axis="both")

    ax_r.plot(np.arange(1, len(cum) + 1), cum * 100, color=SERIES_1, linewidth=2)
    for tau, label_dx in ((0.90, 8), (0.95, 8), (0.99, 8)):
        k = int(np.searchsorted(cum, tau) + 1)
        ax_r.plot([k], [cum[k - 1] * 100], marker="o", markersize=6,
                  color=SERIES_2, markeredgecolor="white", markeredgewidth=1.2,
                  zorder=3)
        ax_r.annotate(f"{int(tau * 100)}%  k={k}", xy=(k, cum[k - 1] * 100),
                      xytext=(label_dx, -14), textcoords="offset points",
                      fontsize=9, color=INK)
    ax_r.set_ylim(0, 104)
    ax_r.set_title("Cumulative explained variance", color=INK, fontsize=12, pad=12)
    ax_r.set_xlabel("components kept", color=INK_MUTED, fontsize=10)
    ax_r.set_ylabel("variance explained (%)", color=INK_MUTED, fontsize=10)
    style_axes(ax_r, grid_axis="both")

    fig.tight_layout()
    _save(fig, out_path)


def plot_criteria(criteria, cum, out_path):
    ordered = sorted(criteria.items(), key=lambda item: item[1])
    labels = [label for label, _ in ordered]
    ks = [k for _, k in ordered]
    # The adopted choice is a state, not a rank, so it gets its own colour.
    colors = [ACCENT if k == CHOSEN_K else SERIES_1 for k in ks]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    fig.patch.set_facecolor("white")

    y = np.arange(len(ks))
    ax.barh(y, ks, color=colors, height=0.62)
    for yi, k in zip(y, ks):
        note = "  <- adopted" if k == CHOSEN_K else ""
        ax.text(k + 3, yi, f"{k}   ({cum[k - 1] * 100:.1f}%){note}",
                va="center", fontsize=9, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(ks) * 1.38)
    ax.set_xlabel("components kept", color=INK_MUTED, fontsize=10)
    ax.set_title("Component-count criteria disagree by a factor of 34",
                 color=INK, fontsize=12, pad=12)
    style_axes(ax, grid_axis="x")

    fig.tight_layout()
    _save(fig, out_path)


def plot_cov_vs_corr(cum_std, cum_raw, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(1, len(cum_std) + 1)
    ax.plot(x, cum_raw * 100, color=SERIES_2, linewidth=2,
            label="covariance (raw, unstandardized)")
    ax.plot(x, cum_std * 100, color=SERIES_1, linewidth=2,
            label="correlation (standardized)")

    ax.axhline(95, color=INK_MUTED, linewidth=1, linestyle="--")
    for cum, color in ((cum_raw, SERIES_2), (cum_std, SERIES_1)):
        k = int(np.searchsorted(cum, 0.95) + 1)
        ax.plot([k], [cum[k - 1] * 100], marker="o", markersize=7, color=color,
                markeredgecolor="white", markeredgewidth=1.4, zorder=3)
        ax.annotate(f"k={k}", xy=(k, cum[k - 1] * 100), xytext=(6, -16),
                    textcoords="offset points", fontsize=10, color=color)

    ax.set_xscale("log")
    ax.set_ylim(0, 104)
    ax.set_xlabel("components kept (log scale)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("variance explained (%)", color=INK_MUTED, fontsize=10)
    ax.set_title("Standardizing spreads variance across more components",
                 color=INK, fontsize=12, pad=12)
    legend = ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK)
    style_axes(ax, grid_axis="both")

    fig.tight_layout()
    _save(fig, out_path)


def plot_correlation_before_after(Z_train, scores, out_path):
    before = np.corrcoef(Z_train[:, :HEATMAP_N], rowvar=False)
    after = np.corrcoef(scores[:, :HEATMAP_N], rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.patch.set_facecolor("white")

    titles = (f"before PCA: first {HEATMAP_N} standardized features",
              f"after PCA: first {HEATMAP_N} component scores")
    for ax, C, title in zip(axes, (before, after), titles):
        image = ax.imshow(C, cmap=CORR_CMAP, vmin=-1, vmax=1)
        ax.set_title(f"{title}\nmean |off-diagonal| = {_mean_abs_offdiag(C):.2e}",
                     color=INK, fontsize=10.5, pad=10)
        ax.set_xticks([0, HEATMAP_N - 1])
        ax.set_yticks([0, HEATMAP_N - 1])
        ax.tick_params(colors=INK_MUTED, length=0)
        for side in ax.spines.values():
            side.set_color(GRID)

    bar = fig.colorbar(image, ax=axes, fraction=0.032, pad=0.03)
    bar.set_label("Pearson correlation", color=INK_MUTED, fontsize=9.5)
    bar.outline.set_color(GRID)
    bar.ax.tick_params(colors=INK_MUTED, length=0)

    _save(fig, out_path)
    return _mean_abs_offdiag(before), _mean_abs_offdiag(after)


def _activity_legend(fig, label_map, ncol=6):
    handles = [
        Line2D([], [], linestyle="none", marker=ACTIVITY_MARKERS[i],
               markersize=7, markerfacecolor=ACTIVITY_COLORS[i],
               markeredgecolor="white", markeredgewidth=0.6,
               label=label_map[i].replace("_", " ").title())
        for i in sorted(label_map)
    ]
    legend = fig.legend(handles=handles, loc="lower center", ncol=ncol,
                        frameon=False, fontsize=9.5,
                        bbox_to_anchor=(0.5, -0.01))
    for text in legend.get_texts():
        text.set_color(INK)


def plot_projection(scores, y_train, label_map, out_path):
    fig = plt.figure(figsize=(13.5, 6))
    fig.patch.set_facecolor("white")

    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")

    for activity in sorted(label_map):
        mask = y_train == activity
        common = dict(color=ACTIVITY_COLORS[activity],
                      marker=ACTIVITY_MARKERS[activity], s=13, alpha=0.55,
                      linewidths=0)
        ax2d.scatter(scores[mask, 0], scores[mask, 1], **common)
        ax3d.scatter(scores[mask, 0], scores[mask, 1], scores[mask, 2], **common)

    ax2d.set_xlabel("PC1", color=INK_MUTED, fontsize=10)
    ax2d.set_ylabel("PC2", color=INK_MUTED, fontsize=10)
    ax2d.set_title("PC1 vs PC2", color=INK, fontsize=12, pad=10)
    style_axes(ax2d, grid_axis="both")

    ax3d.set_xlabel("PC1", color=INK_MUTED, fontsize=9)
    ax3d.set_ylabel("PC2", color=INK_MUTED, fontsize=9)
    ax3d.set_zlabel("PC3", color=INK_MUTED, fontsize=9)
    ax3d.set_title("PC1 / PC2 / PC3", color=INK, fontsize=12, pad=10)
    ax3d.view_init(elev=18, azim=-58)
    # A handful of extreme windows would otherwise shrink the cloud to a dot.
    for axis, limits in zip("xyz", np.percentile(scores[:, :3], [0.5, 99.5], axis=0).T):
        getattr(ax3d, f"set_{axis}lim")(*limits)
    for pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        pane.pane.set_facecolor("white")
        pane.pane.set_edgecolor(GRID)
    ax3d.tick_params(colors=INK_MUTED, labelsize=8)

    _activity_legend(fig, label_map)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out_path)


def plot_biplot(scores, components, feature_names, out_path):
    loadings = components[:2].T
    strength = np.sqrt((loadings**2).sum(axis=1))
    top_plane = list(np.argsort(strength)[::-1][:BIPLOT_ARROWS])
    # PC1's loadings are so evenly spread that nothing reaches the top-8 on the
    # plane, yet PC1 is the axis the report has to explain — so draw its leaders
    # as their own family.
    top_pc1 = [j for j in np.argsort(np.abs(loadings[:, 0]))[::-1][:4]
               if j not in top_plane]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    fig.patch.set_facecolor("white")

    ax.scatter(scores[:, 0], scores[:, 1], s=8, color="#c9c8c3", alpha=0.55,
               linewidths=0)

    # Frame on the bulk, not the handful of extreme windows, so the loadings are
    # readable; a few outlier points fall outside the view.
    (x_lo, x_hi), (y_lo, y_hi) = np.percentile(scores[:, :2], [1, 99], axis=0).T
    pad_x, pad_y = 0.08 * (x_hi - x_lo), 0.10 * (y_hi - y_lo)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

    half = min(max(abs(x_lo), abs(x_hi)), max(abs(y_lo), abs(y_hi)))
    scale = 0.72 * half / np.abs(loadings[top_plane]).max()

    for group, color in ((top_plane, ACCENT), (top_pc1, SERIES_1)):
        for rank, j in enumerate(group):
            dx, dy = loadings[j] * scale
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        linewidth=1.6, shrinkA=0, shrinkB=0))
            # These arrows are near-parallel, so tip labels would pile up.
            # Stagger them in screen space, alternating sides.
            side = 1 if rank % 2 == 0 else -1
            step = 13 * (rank // 2)
            offset = (18 * side, (-8 - step) if dy < 0 else (8 + step))
            ax.annotate(feature_names[j].replace("()", ""), xy=(dx, dy),
                        xytext=offset, textcoords="offset points",
                        fontsize=8.5, color=INK,
                        ha="left" if side > 0 else "right",
                        va="top" if dy < 0 else "bottom")

    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.set_xlabel("PC1", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("PC2", color=INK_MUTED, fontsize=10)
    ax.set_title("Biplot: strongest feature loadings on the PC1/PC2 plane",
                 color=INK, fontsize=12, pad=12)

    handles = [
        Line2D([], [], color=ACCENT, linewidth=2,
               label=f"top {BIPLOT_ARROWS} on the plane"),
        Line2D([], [], color=SERIES_1, linewidth=2, label="top loadings on PC1"),
    ]
    legend = ax.legend(handles=handles, frameon=False, loc="lower left",
                       fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK)
    style_axes(ax, grid_axis="both")

    fig.tight_layout()
    _save(fig, out_path)
    return ([feature_names[j] for j in top_plane],
            [feature_names[j] for j in top_pc1])


def reconstruction_table(Z_train, pca, scores, out_csv):
    p = Z_train.shape[1]
    eigvals = pca.explained_variance_
    rows = []
    for k in RECON_KS:
        recon = _reconstruct(pca.mean_, pca.components_, scores, k)
        empirical = float(((Z_train - recon) ** 2).mean())
        theoretical = float(eigvals[k:].sum() / p)
        rows.append({
            "k": k,
            "variance_retained_pct": float(pca.explained_variance_ratio_[:k].sum() * 100),
            "empirical_mse": empirical,
            "theoretical_discarded_variance": theoretical,
            "abs_difference": abs(empirical - theoretical),
        })

    frame = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    print(f"  saved {out_csv.name}")
    return frame


def plot_reconstruction(frame, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.patch.set_facecolor("white")

    ax.plot(frame["k"], frame["theoretical_discarded_variance"], color=SERIES_1,
            linewidth=2.4, label=r"theoretical  $\sum_{i>k}\lambda_i\,/\,p$")
    ax.plot(frame["k"], frame["empirical_mse"], color=SERIES_2, linewidth=0,
            marker="o", markersize=6, markeredgecolor="white",
            markeredgewidth=1.2, label="empirical reconstruction MSE")

    ax.axvline(CHOSEN_K, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.annotate(f"k={CHOSEN_K}", xy=(CHOSEN_K, frame["empirical_mse"].max()),
                xytext=(6, -4), textcoords="offset points", fontsize=9,
                color=INK_MUTED)

    ax.set_xscale("log")
    ax.set_xlabel("components kept (log scale)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("mean squared error", color=INK_MUTED, fontsize=10)
    ax.set_title("Truncation error equals the variance thrown away",
                 color=INK, fontsize=12, pad=12)
    legend = ax.legend(frameon=False, fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(INK)
    style_axes(ax, grid_axis="both")

    fig.tight_layout()
    _save(fig, out_path)


def run(out_dir=FIGURES_DIR, results_dir=RESULTS_DIR):
    out_dir = Path(out_dir)
    results_dir = Path(results_dir)

    X_train, X_test, y_train, _, _, _, feature_names, label_map = load_har()
    Z_train, _, _ = standardize(X_train, X_test)

    pca = MyPCA().fit(Z_train)
    eigvals = pca.explained_variance_
    cum = np.cumsum(pca.explained_variance_ratio_)
    scores = pca.transform(Z_train)

    raw = MyPCA().fit(X_train)
    cum_raw = np.cumsum(raw.explained_variance_ratio_)

    criteria = all_criteria(eigvals, pca.explained_variance_ratio_)
    elbow = k_scree_elbow(eigvals)

    print("figures:")
    plot_scree_and_cumulative(eigvals, cum, elbow, out_dir / "02_scree_and_cumulative.png")
    plot_criteria(criteria, cum, out_dir / "03_criteria_comparison.png")
    plot_cov_vs_corr(cum, cum_raw, out_dir / "04_cov_vs_corr.png")
    before, after = plot_correlation_before_after(
        Z_train, scores, out_dir / "05_correlation_before_after.png")
    plot_projection(scores, y_train, label_map, out_dir / "06_projection_2d_3d.png")
    top_plane, top_pc1 = plot_biplot(scores, pca.components_, feature_names,
                                     out_dir / "07_biplot.png")

    frame = reconstruction_table(Z_train, pca, scores,
                                 results_dir / "reconstruction_error.csv")
    plot_reconstruction(frame, out_dir / "08_reconstruction_error.png")

    print(f"\nmean |off-diagonal correlation|, first {HEATMAP_N}:")
    print(f"  before PCA  {before:.4f}")
    print(f"  after PCA   {after:.3e}")

    print(f"\ntop {BIPLOT_ARROWS} loadings on the PC1/PC2 plane:")
    for name in top_plane:
        print(f"  {name}")
    print("top loadings on PC1 alone:")
    for name in top_pc1:
        print(f"  {name}")

    print("\nreconstruction error vs k:")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    return frame, (before, after)


if __name__ == "__main__":
    run()
