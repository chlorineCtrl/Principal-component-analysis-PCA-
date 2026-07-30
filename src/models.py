"""
four classifiers on the 561-dim standardized features and on the
102-dim PCA scores, timed and scored the same way.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_data import load_har
from preprocess import standardize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

CHOSEN_K = 102
RANDOM_STATE = 42
CV_SPLITS = 3


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  random_state=RANDOM_STATE),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale",
                         random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100,
                                                random_state=RANDOM_STATE,
                                                n_jobs=-1),
    }


def make_pipeline(estimator, n_components=CHOSEN_K):
    """Scaler + PCA + classifier as one estimator.

    Anything cross-validated must go through this: the scaler and the PCA are
    refit inside each training fold, so a fold's held-out rows never influence
    the transform applied to them.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("clf", clone(estimator)),
    ])


def evaluate(estimator, X_train, y_train, X_test, y_test):
    """Fit and predict once, timing both."""
    model = clone(estimator)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def run_comparison(Z_train, Z_test, y_train, y_test, k=CHOSEN_K):
    """Every model on both representations. Returns the tidy long-form table."""
    pca = PCA(n_components=k, random_state=RANDOM_STATE).fit(Z_train)
    P_train = pca.transform(Z_train)
    P_test = pca.transform(Z_test)

    spaces = {
        f"before ({Z_train.shape[1]}-dim)": (Z_train, Z_test),
        f"after ({k}-dim)": (P_train, P_test),
    }

    rows = []
    for name, estimator in build_models().items():
        for space, (A_train, A_test) in spaces.items():
            scores = evaluate(estimator, A_train, y_train, A_test, y_test)
            rows.append({"model": name, "space": space, **scores})
            print(f"  {name:<20} {space:<18} acc={scores['accuracy']:.4f} "
                  f"fit={scores['fit_time_s']:.2f}s")

    return pd.DataFrame(rows), pca


def summarize(long_frame):
    """Collapse the long table into one row per model: before, after, deltas."""
    before_key, after_key = long_frame["space"].unique()
    before = long_frame[long_frame["space"] == before_key].set_index("model")
    after = long_frame[long_frame["space"] == after_key].set_index("model")

    total_before = before["fit_time_s"] + before["predict_time_s"]
    total_after = after["fit_time_s"] + after["predict_time_s"]

    summary = pd.DataFrame({
        "acc_before": before["accuracy"],
        "acc_after": after["accuracy"],
        "change_pp": (after["accuracy"] - before["accuracy"]) * 100,
        "macro_f1_before": before["macro_f1"],
        "macro_f1_after": after["macro_f1"],
        "fit_time_before_s": before["fit_time_s"],
        "fit_time_after_s": after["fit_time_s"],
        "predict_time_before_s": before["predict_time_s"],
        "predict_time_after_s": after["predict_time_s"],
        "speedup_total": total_before / total_after,
    })
    return summary.sort_values("change_pp").reset_index()


def cross_validate_pipeline(X_train, y_train, groups, estimator, k=CHOSEN_K):
    """Leakage-safe CV: folds split by subject, transforms refit per fold."""
    pipeline = make_pipeline(estimator, k)
    splitter = GroupKFold(n_splits=CV_SPLITS)
    return cross_val_score(pipeline, X_train, y_train, groups=groups,
                           cv=splitter, scoring="accuracy", n_jobs=1)


def run(results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    X_train, X_test, y_train, y_test, subj_train, _, _, _ = load_har()
    Z_train, Z_test, _ = standardize(X_train, X_test)

    print("fitting models:")
    long_frame, _ = run_comparison(Z_train, Z_test, y_train, y_test)
    summary = summarize(long_frame)

    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "model_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv.name}")

    print(f"\nmodel                acc before  acc after   change    speed-up")
    for row in summary.itertuples():
        print(f"  {row.model:<20} {row.acc_before:.4f}     {row.acc_after:.4f}   "
              f"{row.change_pp:+.2f}pp     {row.speedup_total:.1f}x")

    scores = cross_validate_pipeline(X_train, y_train, subj_train,
                                     build_models()["SVM (RBF)"])
    print(f"\nleakage-safe CV (SVM in a Pipeline, {CV_SPLITS}-fold GroupKFold "
          f"by subject):")
    print(f"  fold accuracies {np.round(scores, 4)}")
    print(f"  mean {scores.mean():.4f}  sd {scores.std():.4f}")

    return summary


if __name__ == "__main__":
    run()
