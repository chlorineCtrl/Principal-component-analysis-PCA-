"""
Step 5 — PCA from first principles, via the eigendecomposition of the covariance
matrix. Sigma = Xc.T @ Xc / (n - 1); the components are its eigenvectors and the
variance each captures is the matching eigenvalue.
"""

import numpy as np


class MyPCA:
    """PCA by eigendecomposition. Rows of components_ are the components."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        cov = (Xc.T @ Xc) / (n - 1)

        # eigh, not eig: cov is real symmetric, so this is faster and returns
        # real eigenvalues instead of complex ones carrying rounding noise.
        eigvals, eigvecs = np.linalg.eigh(cov)

        # eigh returns them ascending. PCA needs largest variance first.
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # A covariance matrix cannot have negative eigenvalues; anything below
        # zero here is float error on a near-singular matrix.
        eigvals = np.clip(eigvals, 0.0, None)

        total = eigvals.sum()
        self.explained_variance_ = eigvals
        self.explained_variance_ratio_ = eigvals / total
        self.components_ = eigvecs.T

        if self.n_components is not None:
            k = self.n_components
            self.explained_variance_ = self.explained_variance_[:k]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:k]
            self.components_ = self.components_[:k]

        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=np.float64) @ self.components_ + self.mean_


def run():
    from preprocess import standardize
    from load_data import load_har

    X_train, X_test, *_ = load_har()
    Z_train, _, _ = standardize(X_train, X_test)

    pca = MyPCA().fit(Z_train)
    eigvals = pca.explained_variance_

    print(f"fitted on {Z_train.shape}")
    print(f"components_ shape {pca.components_.shape}")
    print(f"\ntotal variance captured  {eigvals.sum():.4f}   "
          f"(= p = {Z_train.shape[1]} for standardized features)")
    print(f"first five eigenvalues   {np.round(eigvals[:5], 4)}")
    print(f"first five evr           {np.round(pca.explained_variance_ratio_[:5], 4)}")
    print(f"smallest eigenvalue      {eigvals[-1]:.3e}")

    # Orthonormality and the round-trip identity, both implied by the derivation.
    gram = pca.components_ @ pca.components_.T
    print(f"\nmax |W W^T - I|          {np.abs(gram - np.eye(len(gram))).max():.3e}")
    scores = pca.transform(Z_train)
    recon = pca.inverse_transform(scores)
    print(f"max |X - recon| (all PCs) {np.abs(Z_train - recon).max():.3e}")

    # Cov(scores) must be diagonal with the eigenvalues on it.
    score_cov = np.cov(scores[:, :5], rowvar=False)
    off_diag = score_cov - np.diag(np.diag(score_cov))
    print(f"max |off-diagonal Cov(scores)| {np.abs(off_diag).max():.3e}")

    return pca


if __name__ == "__main__":
    run()
