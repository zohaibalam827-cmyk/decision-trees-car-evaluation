"""Evaluate ID3 honestly: train on one part of the data, test on cars it has never seen."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dtrees.id3 import ID3DecisionTree
from dtrees.utils import accuracy_score, train_test_split

cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data/car.data", names=cols)
X, y = df.drop(columns="class"), df["class"].to_numpy()

# ---- one split: 70% train / 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42)
print("train rows:", len(X_train), "| test rows:", len(X_test))

baseline = np.mean(y_test == "unacc")  # a "model" that always guesses the biggest class
tree = ID3DecisionTree().fit(X_train, y_train)
pred = tree.predict(X_test)
print(f"baseline (always 'unacc'): {baseline:.3f}")
print(f"ID3 train accuracy:        {tree.score(X_train, y_train):.3f}")
print(f"ID3 test accuracy:         {accuracy_score(y_test, pred):.3f}")
print()
print(pd.crosstab(pd.Series(y_test, name="actual"), pd.Series(pred, name="predicted")))

# ---- repeat with 10 different random splits
scores = []
for seed in range(10):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, seed=seed)
    scores.append(ID3DecisionTree().fit(Xtr, ytr).score(Xte, yte))
print()
print(f"10 splits: mean={np.mean(scores):.3f}  std={np.std(scores):.3f}  "
      f"min={min(scores):.3f}  max={max(scores):.3f}")
