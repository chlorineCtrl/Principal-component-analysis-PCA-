import time
from pathlib import Path

import numpy as np

import component_selection
import confusion_analysis
import download_data
import eda
import models
import preprocess
import sweep_k
import validate_pca
import visualize
from load_data import load_har
from pca_scratch import MyPCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

CHOSEN_K = 102


def banner(step, title):
    print(f"\n{'=' * 72}")
    print(f"STEP {step} — {title}")
    print("=" * 72)


def main():
    started = time.perf_counter()
    timings = {}

    def timed(step, title, function, *args, **kwargs):
        banner(step, title)
        start = time.perf_counter()
        result = function(*args, **kwargs)
        timings[title] = time.perf_counter() - start
        return result

    timed(1, "download", download_data.ensure_dataset)

    banner(2, "load")
    start = time.perf_counter()
    (X_train, X_test, y_train, y_test,
     subj_train, subj_test, feature_names, label_map) = load_har()
    print(f"X_train {X_train.shape}   X_test {X_test.shape}")
    print(f"subject overlap: {set(subj_train) & set(subj_test)}")
    timings["load"] = time.perf_counter() - start

    timed(3, "eda", eda.run)
    Z_train, Z_test, _ = timed(4, "preprocess", preprocess.run)

    banner(5, "PCA from scratch")
    start = time.perf_counter()
    pca = MyPCA().fit(Z_train)
    eigvals = pca.explained_variance_
    print(f"total variance {eigvals.sum():.4f}  "
          f"first five {np.round(eigvals[:5], 4)}")
    timings["pca"] = time.perf_counter() - start

    diffs, d_svd = timed(6, "validate", validate_pca.run)
    criteria, cum, k_raw = timed(7, "component selection", component_selection.run)
    recon, (corr_before, corr_after) = timed(8, "visualize", visualize.run)
    summary = timed(9, "models", models.run)
    sweep = timed(10, "sweep k", sweep_k.run)
    report, cm_before, cm_after = timed(11, "confusion analysis",
                                        confusion_analysis.run)

    elapsed = time.perf_counter() - started
    figures = sorted(FIGURES_DIR.glob("*.png"))
    csvs = sorted(RESULTS_DIR.glob("*.csv"))

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print("=" * 72)

    print(f"\ndataset      train {X_train.shape}, test {X_test.shape}, "
          f"{len(label_map)} classes, {len(feature_names)} features")
    print(f"split        {len(set(subj_train))} train subjects / "
          f"{len(set(subj_test))} test subjects, overlap "
          f"{set(subj_train) & set(subj_test)}")

    print(f"\nMyPCA vs sklearn   max |diff| eigenvalues {diffs['eigenvalues']:.2e}, "
          f"components {diffs['components (up to sign)']:.2e}, "
          f"scores {diffs['scores (signs aligned)']:.2e}")
    print(f"SVD identity       max |lambda - s^2/(n-1)| {d_svd:.2e}")

    print(f"\ncomponent counts by criterion:")
    for label, k in sorted(criteria.items(), key=lambda item: item[1]):
        mark = "  <- adopted" if k == CHOSEN_K else ""
        print(f"  {label:<20} k={k:<5} ({cum[k - 1] * 100:.1f}% variance){mark}")
    print(f"  raw covariance needs k={k_raw} for 95% (vs "
          f"k={criteria['95% variance']} standardized)")

    print(f"\ncorrelation among the first 30 dimensions:")
    print(f"  before PCA  mean |off-diagonal| {corr_before:.4f}")
    print(f"  after PCA   mean |off-diagonal| {corr_after:.2e}")

    chosen_recon = recon.loc[recon["k"] == CHOSEN_K].iloc[0]
    print(f"reconstruction at k={CHOSEN_K}: empirical MSE "
          f"{chosen_recon['empirical_mse']:.6f} vs theoretical "
          f"{chosen_recon['theoretical_discarded_variance']:.6f}")

    print(f"\nmodel                acc before  acc after   change    speed-up")
    for row in summary.itertuples():
        print(f"  {row.model:<20} {row.acc_before:.4f}     {row.acc_after:.4f}   "
              f"{row.change_pp:+.2f}pp     {row.speedup_total:.1f}x")

    best = sweep.loc[sweep["test_accuracy"].idxmax()]
    chosen = sweep.loc[sweep["k"] == CHOSEN_K].iloc[0]
    print(f"\nk sweep (SVM-RBF)  best k={int(best['k'])} at "
          f"{best['test_accuracy']:.4f}; k={CHOSEN_K} at "
          f"{chosen['test_accuracy']:.4f} "
          f"({(best['test_accuracy'] - chosen['test_accuracy']) * 100:.2f}pp lower)")

    worst = confusion_analysis.worst_pairs(cm_after,
                                           [label_map[i] for i in sorted(label_map)],
                                           top=1)[0]
    print(f"largest confusion  {worst[1]} <-> {worst[2]}: {worst[0]} windows")
    print(f"LAYING recall      {report.loc['LAYING', 'recall']:.4f}")

    print(f"\noutputs      {len(figures)} figures, {len(csvs)} csvs")
    for path in figures:
        print(f"  figures/{path.name}")
    for path in csvs:
        print(f"  results/{path.name}")

    print(f"\nstep timings:")
    for label, seconds in timings.items():
        print(f"  {label:<22} {seconds:6.2f}s")
    print(f"\ntotal {elapsed:.1f}s")


if __name__ == "__main__":
    main()
