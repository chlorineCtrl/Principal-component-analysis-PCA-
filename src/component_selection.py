"""
how many components to keep. Five unsupervised criteria, all computed
from the same full 561-component fit, plus a raw-vs-standardized comparison.
"""

import numpy as np

from load_data import load_har
from preprocess import standardize
from pca_scratch import MyPCA

SCREE_WINDOW = 60


def k_for_variance(evr, tau):
    """Smallest k whose cumulative explained variance reaches tau."""
    return int(np.searchsorted(np.cumsum(evr), tau) + 1)


def k_kaiser(eigvals):
    """Count of eigenvalues above 1. Only meaningful on standardized data,
    where the mean eigenvalue is exactly 1, so 'above average' means lambda > 1."""
    return int((eigvals > 1.0).sum())


def broken_stick_expected(p):
    """b_i = (1/p) * sum_{j>=i} 1/j — expected share of piece i when a unit stick
    is broken at random into p pieces."""
    inv = 1.0 / np.arange(1, p + 1)
    return np.cumsum(inv[::-1])[::-1] / p


def k_broken_stick(evr):
    """Keep components while they beat random; k is the first index that fails."""
    expected = broken_stick_expected(len(evr))
    failed = np.flatnonzero(evr < expected)
    return int(failed[0]) if failed.size else len(evr)


def k_scree_elbow(eigvals, window=SCREE_WINDOW):
    """Point of maximum perpendicular distance to the chord joining the first
    and last eigenvalue in the window."""
    y = eigvals[:window]
    x = np.arange(len(y), dtype=float)
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])

    chord = end - start
    points = np.column_stack([x, y]) - start
    # 2-D cross product gives the area of the parallelogram, divide by the base
    # length to get the perpendicular height. (numpy 2 no longer takes the cross product of 2-D vectors.)
    area = np.abs(chord[0] * points[:, 1] - chord[1] * points[:, 0])
    distance = area / np.linalg.norm(chord)
    return int(np.argmax(distance) + 1)


def all_criteria(eigvals, evr):
    """Every criterion as {label: k}."""
    criteria = {
        "Scree elbow": k_scree_elbow(eigvals),
        "Broken stick": k_broken_stick(evr),
        "Kaiser (lambda>1)": k_kaiser(eigvals),
    }
    for tau in (0.80, 0.90, 0.95, 0.99):
        criteria[f"{int(tau * 100)}% variance"] = k_for_variance(evr, tau)
    return criteria


def run():
    X_train, X_test, *_ = load_har()
    Z_train, _, _ = standardize(X_train, X_test)

    pca = MyPCA().fit(Z_train)
    eigvals = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    criteria = all_criteria(eigvals, evr)

    print("criterion               k       variance captured")
    for label, k in sorted(criteria.items(), key=lambda item: item[1]):
        print(f"  {label:<20} k={k:<5}  ({cum[k - 1] * 100:.1f}%)")

    # Same data without standardizing: PCA on the raw covariance matrix rather than the correlation matrix.
    raw = MyPCA().fit(X_train)
    raw_evr = raw.explained_variance_ratio_
    k_raw = k_for_variance(raw_evr, 0.95)
    k_std = criteria["95% variance"]

    print(f"\ncovariance matrix (raw, unstandardized) needs k={k_raw} for 95% variance")
    print(f"correlation matrix (standardized)       needs k={k_std} for 95% variance")
    print(f"PC1 alone: raw={raw_evr[0] * 100:.1f}% variance, "
          f"standardized={evr[0] * 100:.1f}% variance")

    return criteria, cum, k_raw


if __name__ == "__main__":
    run()
