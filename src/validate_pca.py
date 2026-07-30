"""
prove MyPCA against sklearn's PCA and against a manual SVD, so the
derivation in Step 5 is confirmed numerically .
"""

import numpy as np
from sklearn.decomposition import PCA

from load_data import load_har
from preprocess import standardize
from pca_scratch import MyPCA

K = 50
TOL_EIG = 1e-8
TOL_COMP = 1e-8
TOL_SCORE = 1e-6


def compare_with_sklearn(Z, k=K):
    """Max absolute differences between MyPCA and sklearn over the first k PCs."""
    mine = MyPCA().fit(Z)
    theirs = PCA(svd_solver="full").fit(Z)

    d_eig = np.abs(mine.explained_variance_[:k] - theirs.explained_variance_[:k]).max()
    d_evr = np.abs(
        mine.explained_variance_ratio_[:k] - theirs.explained_variance_ratio_[:k]
    ).max()

    
    W_mine = mine.components_[:k]
    W_theirs = theirs.components_[:k]
    d_comp = np.abs(np.abs(W_mine) - np.abs(W_theirs)).max()

    
    sign = np.sign(np.sum(W_mine * W_theirs, axis=1))
    d_score = np.abs(mine.transform(Z)[:, :k] * sign - theirs.transform(Z)[:, :k]).max()

    return mine, {
        "eigenvalues": d_eig,
        "explained variance ratio": d_evr,
        "components (up to sign)": d_comp,
        "scores (signs aligned)": d_score,
        "flipped components": int((sign < 0).sum()),
    }


def compare_with_svd(Z, mine):
    """Check lambda_i = s_i^2 / (n - 1) directly from the SVD of centred data."""
    Zc = Z - Z.mean(axis=0)
    _, S, _ = np.linalg.svd(Zc, full_matrices=False)
    lam_from_svd = S**2 / (Z.shape[0] - 1)
    return np.abs(mine.explained_variance_ - lam_from_svd).max()


def run():
    X_train, X_test, *_ = load_har()
    Z_train, _, _ = standardize(X_train, X_test)

    mine, diffs = compare_with_sklearn(Z_train)
    flipped = diffs.pop("flipped components")

    print(f"MyPCA vs sklearn.decomposition.PCA, first {K} components:")
    for label, value in diffs.items():
        print(f"  max |difference| in {label:<26}: {value:.3e}")
    print(f"  components sign-flipped vs sklearn      : {flipped} of {K}")

    d_svd = compare_with_svd(Z_train, mine)
    print(f"\n  max |lambda_eig - s^2/(n-1)| (SVD check) : {d_svd:.3e}")

    assert diffs["eigenvalues"] < TOL_EIG, diffs["eigenvalues"]
    assert diffs["explained variance ratio"] < TOL_EIG, diffs["explained variance ratio"]
    assert diffs["components (up to sign)"] < TOL_COMP, diffs["components (up to sign)"]
    assert diffs["scores (signs aligned)"] < TOL_SCORE, diffs["scores (signs aligned)"]
    assert d_svd < TOL_EIG, d_svd

    print("\nPASSED")
    return diffs, d_svd


if __name__ == "__main__":
    run()
