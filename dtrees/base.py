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


# ---------------------------------------------------------------- Part C1: training side
import pandas as pd


class BaseDecisionTree:
    """Common logic. ID3, C4.5 and CART only have to write their own _build()."""

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if X.isnull().any().any():
            raise ValueError("Missing values are not supported - fill them first")
        self.feature_names_ = list(X.columns)
        self.is_numeric_ = {c: bool(pd.api.types.is_numeric_dtype(X[c])) for c in X.columns}
        self.classes_ = np.unique(y)
        self._validate_features()
        self.root_ = self._build(X, y, 0, list(self.feature_names_))
        self._post_fit()
        return self

    def _validate_features(self):
        pass  # ID3 overrides this to reject numeric columns

    def _post_fit(self):
        pass  # C4.5 and CART override this to prune the tree

    def _build(self, X, y, depth, available):
        raise NotImplementedError("each algorithm writes its own _build")

    @staticmethod
    def _make_node(y):
        classes, counts = np.unique(y, return_counts=True)
        return Node(
            prediction=classes.tolist()[int(np.argmax(counts))],
            class_counts=dict(zip(classes.tolist(), counts.tolist())),
            n_samples=int(len(y)),
        )

    def _stop(self, node, depth):
        """True when this node must NOT be split any further."""
        return (
            node.errors == 0
            or node.n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        )

    # ------------------------------------------------------ Part C2: predict and inspect
    def predict(self, X):
        X = pd.DataFrame(X)
        return np.array([self._predict_row(row) for row in X.to_dict("records")])

    def _predict_row(self, row):
        node = self.root_
        while node.children:
            value = row[node.feature]
            if node.kind == "multiway":
                nxt = node.children.get(value)
                if nxt is None:  # a value never seen in training here: stop and guess
                    break
            elif node.kind == "numeric":
                nxt = node.children["<="] if value <= node.threshold else node.children[">"]
            else:  # "subset"
                nxt = node.children["in"] if value in node.threshold else node.children["out"]
            node = nxt
        return node.prediction

    def score(self, X, y):
        """Accuracy: fraction of rows predicted correctly."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @staticmethod
    def _leaves(node):
        if node.is_leaf:
            return [node]
        return [leaf for c in node.children.values() for leaf in BaseDecisionTree._leaves(c)]

    @staticmethod
    def _internal_nodes(node):
        if node.is_leaf:
            return []
        found = [node]
        for c in node.children.values():
            found += BaseDecisionTree._internal_nodes(c)
        return found

    def n_leaves(self):
        return len(self._leaves(self.root_))

    def depth(self):
        def d(node):
            return 0 if node.is_leaf else 1 + max(d(c) for c in node.children.values())
        return d(self.root_)

    def export_text(self):
        """Return the tree as indented text."""
        def condition(node, key):
            if node.kind == "multiway":
                return f"{node.feature} = {key}"
            if node.kind == "numeric":
                return f"{node.feature} {key} {node.threshold:g}"
            word = "in" if key == "in" else "not in"
            return f"{node.feature} {word} {sorted(node.threshold, key=str)}"

        lines = []

        def walk(node, level):
            pad = "|   " * level
            for key, child in node.children.items():
                if child.is_leaf:
                    lines.append(f"{pad}|--- {condition(node, key)}  ->  {child.prediction}  {child.class_counts}")
                else:
                    lines.append(f"{pad}|--- {condition(node, key)}")
                    walk(child, level + 1)

        if self.root_.is_leaf:
            return f"->  {self.root_.prediction}  {self.root_.class_counts}"
        walk(self.root_, 0)
        return "\n".join(lines)
