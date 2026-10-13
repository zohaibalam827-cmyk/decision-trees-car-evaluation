"""Compare ID3, C4.5 without pruning and C4.5 with pruning on unseen cars."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dtrees.c45 import C45DecisionTree
from dtrees.id3 import ID3DecisionTree
from dtrees.utils import train_test_split

cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv("data/car.data", names=cols)
X, y = df.drop(columns="class"), df["class"].to_numpy()

models = {
    "ID3": lambda: ID3DecisionTree(),
    "C4.5 no pruning": lambda: C45DecisionTree(prune=False),
    "C4.5 pruned": lambda: C45DecisionTree(prune=True),
}

results = {name: {"test": [], "leaves": []} for name in models}
for seed in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=seed)
    for name, make in models.items():
        tree = make().fit(X_train, y_train)
        results[name]["test"].append(tree.score(X_test, y_test))
        results[name]["leaves"].append(tree.n_leaves())

print(f"{'model':18s} {'test acc (mean)':>16s} {'std':>7s} {'leaves (mean)':>14s}")
for name, r in results.items():
    print(f"{name:18s} {np.mean(r['test']):16.3f} {np.std(r['test']):7.3f} {np.mean(r['leaves']):14.1f}")
