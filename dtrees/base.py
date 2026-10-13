"""Shared building blocks for ID3, C4.5 and CART."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------- Part A: impurity
def entropy_from_counts(counts):
    """Entropy (base 2) from class counts, e.g. [9, 5]."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts[counts > 0] / n
    return max(0.0, float(-(p * np.log2(p)).sum()))


def gini_from_counts(counts):
    """Gini impurity from class counts, e.g. [9, 5]."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return float(1.0 - (p ** 2).sum())


def entropy(y):
    """Entropy of a list/array of class labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return entropy_from_counts(counts)


def gini(y):
    """Gini impurity of a list/array of class labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(np.asarray(y), return_counts=True)
    return gini_from_counts(counts)


def partition_masks(col):
    """For a categorical column, return {value: boolean mask of the rows with that value}."""
    return {v: col == v for v in sorted(set(col.tolist()), key=str)}


# ---------------------------------------------------------------- Part B: the Node
@dataclass
class Node:
    prediction: Any                        # majority class at this node
    class_counts: Dict[Any, int]           # e.g. {"unacc": 10, "acc": 2}
    n_samples: int                         # training rows that reached this node
    feature: Optional[str] = None          # feature tested here (None for a leaf)
    kind: Optional[str] = None             # "multiway", "numeric" or "subset"
    threshold: Any = None                  # used by numeric / subset splits
    children: Dict[Any, "Node"] = field(default_factory=dict)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def errors(self):
        """Rows misclassified if this node were a leaf."""
        return self.n_samples - max(self.class_counts.values())

    def collapse(self):
        """Turn this node into a leaf (used by pruning)."""
        self.children = {}
        self.feature = self.kind = self.threshold = None
