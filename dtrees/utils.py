"""Small helpers: splitting data and measuring accuracy."""
import numpy as np
import pandas as pd


def train_test_split(X, y, test_size=0.3, seed=42):
    """Shuffle, then split. Each class is split separately (stratified), so the
    train and test parts both keep roughly the same class proportions."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    test_rows = []
    for cls in np.unique(y):
        rows = rng.permutation(np.where(y == cls)[0])
        test_rows += rows[: max(1, int(round(len(rows) * test_size)))].tolist()

    is_test = np.zeros(len(y), dtype=bool)
    is_test[test_rows] = True
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


def accuracy_score(y_true, y_pred):
    """Fraction of predictions that are correct."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
