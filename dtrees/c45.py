"""C4.5 (Quinlan, 1993): gain ratio, numeric features, pruning."""
import math
from statistics import NormalDist  # used by the pruning code in Part 2

import numpy as np

from .base import BaseDecisionTree, entropy, entropy_from_counts, partition_masks


class C45DecisionTree(BaseDecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=2,
                 prune=True, confidence=0.25):
        super().__init__(max_depth, min_samples_split)
        self.min_samples_leaf = min_samples_leaf  # each side of a numeric split needs this many rows
        self.prune = prune                        # switch pruning on/off
        self.confidence = confidence              # pruning strength: smaller = more pruning

    # ------------------------------------------------------------ Part 1: building
    def _build(self, X, y, depth, available):
        node = self._make_node(y)
        if self._stop(node, depth):
            return node
        n, base = len(y), entropy(y)

        # Every candidate split is a tuple: (feature, kind, threshold, gain, split_info)
        candidates = []
        for f in self.feature_names_:
            col = X[f].to_numpy()
            if self.is_numeric_[f]:
                found = self._best_numeric_split(col.astype(float), y, base)
                if found:
                    gain, split_info, threshold = found
                    candidates.append((f, "numeric", threshold, gain, split_info))
            elif f in available:
                masks = partition_masks(col)
                if len(masks) < 2:
                    continue
                weights = np.array([m.sum() for m in masks.values()]) / n
                gain = base - sum(w * entropy(y[m]) for w, m in zip(weights, masks.values()))
                split_info = float(-(weights * np.log2(weights)).sum())
                candidates.append((f, "multiway", None, gain, split_info))

        if not candidates:
            return node

        # Quinlan's rule: only features with at-least-average gain compete on gain ratio
        average_gain = np.mean([c[3] for c in candidates])
        good = [c for c in candidates if c[3] >= average_gain - 1e-12 and c[4] > 0]
        if not good:
            return node
        feature, kind, threshold, gain, split_info = max(good, key=lambda c: c[3] / c[4])
        if gain <= 1e-12:
            return node

        node.feature, node.kind, node.threshold = feature, kind, threshold
        col = X[feature].to_numpy()
        if kind == "numeric":
            left = col.astype(float) <= threshold
            node.children["<="] = self._build(X[left], y[left], depth + 1, available)
            node.children[">"] = self._build(X[~left], y[~left], depth + 1, available)
        else:
            remaining = [a for a in available if a != feature]
            for value, mask in partition_masks(col).items():
                node.children[value] = self._build(X[mask], y[mask], depth + 1, remaining)
        return node

    def _best_numeric_split(self, col, y, base):
        """Try 'col <= t' for every useful t. Return (gain, split_info, t) or None."""
        n = len(y)
        order = np.argsort(col, kind="mergesort")
        sorted_x, sorted_y = col[order], y[order]
        onehot = np.zeros((n, len(self.classes_)))
        onehot[np.arange(n), np.searchsorted(self.classes_, sorted_y)] = 1
        left_counts = np.cumsum(onehot, axis=0)  # class counts of the first i+1 rows
        total = left_counts[-1]

        best = None
        for i in range(n - 1):
            if sorted_x[i] == sorted_x[i + 1]:
                continue  # can't split between two equal values
            n_left, n_right = i + 1, n - i - 1
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            lc = left_counts[i]
            child = (n_left / n) * entropy_from_counts(lc) + (n_right / n) * entropy_from_counts(total - lc)
            gain = base - child
            if best is None or gain > best[0]:
                split_info = -((n_left / n) * math.log2(n_left / n) + (n_right / n) * math.log2(n_right / n))
                best = (gain, split_info, float(sorted_x[i]))  # threshold = a real data value
        return best

    # ------------------------------------------------------------ Part 2: pruning
    def _post_fit(self):
        if self.prune:
            self._z = NormalDist().inv_cdf(1 - self.confidence)
            self._prune_node(self.root_)

    def _estimated_errors(self, n, errors):
        """Pessimistic guess of how many of n rows this node will get wrong on NEW data."""
        if n == 0:
            return 0.0
        if errors == 0:
            return n * (1 - self.confidence ** (1 / n))
        z, f = self._z, errors / n
        upper = (f + z * z / (2 * n)
                 + z * math.sqrt(f / n - f * f / n + z * z / (4 * n * n))) / (1 + z * z / n)
        return n * upper

    def _prune_node(self, node):
        """Bottom-up. Returns the estimated errors of this (possibly pruned) subtree."""
        as_leaf = self._estimated_errors(node.n_samples, node.errors)
        if node.is_leaf:
            return as_leaf
        as_subtree = sum(self._prune_node(child) for child in node.children.values())
        if as_leaf <= as_subtree + 0.1:  # a plain leaf is about as good: cut the subtree
            node.collapse()
            return as_leaf
        return as_subtree
