"""
Step 4 — standardize the features, fitting the scaler on train only.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from load_data import load_har


def standardize(X_train, X_test):
    """Fit StandardScaler on train, transform both. Returns (Ztr, Zte, scaler)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler


def summarize(Z_train, Z_test):
    """Print the global and per-feature moments used as this step's check."""
    for name, Z in (("train", Z_train), ("test ", Z_test)):
        print(f"{name}: mean={Z.mean():+.3e}  std={Z.std(ddof=0):.6f}")

    # The global mean averages over 561 columns, so opposite shifts cancel and
    # the number looks smaller than the actual per-feature drift.
    print("\nper-feature deviation from the train fit:")
    for name, Z in (("train", Z_train), ("test ", Z_test)):
        col_mean = Z.mean(axis=0)
        col_std = Z.std(axis=0, ddof=0)
        print(f"  {name}: max|mean|={np.abs(col_mean).max():.4f}   "
              f"std range [{col_std.min():.4f}, {col_std.max():.4f}]")


def leakage_check(X_test):
    """What the test moments would look like if the scaler were refit on test."""
    Z = StandardScaler().fit_transform(X_test)
    print("\nif the scaler had been (wrongly) refit on test:")
    print(f"  test : mean={Z.mean():+.3e}  std={Z.std(ddof=0):.6f}   <- exactly 0/1")


def run():
    X_train, X_test, *_ = load_har()
    Z_train, Z_test, scaler = standardize(X_train, X_test)

    print(f"X_train {X_train.shape}   X_test {X_test.shape}\n")
    summarize(Z_train, Z_test)
    leakage_check(X_test)

    return Z_train, Z_test, scaler


if __name__ == "__main__":
    run()
