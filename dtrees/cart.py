"""CART (Breiman, Friedman, Olshen & Stone, 1984): Gini impurity, binary splits, pruning."""
from itertools import combinations

import numpy as np

from .base import BaseDecisionTree, gini, gini_from_counts


class CARTDecisionTree(BaseDecisionTree):
    MAX_EXHAUSTIVE_VALUES = 10  # try every category grouping up to this many values

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):
        super().__init__(max_depth, min_samples_split)
        self.min_samples_leaf = min_samples_leaf              # every leaf keeps at least this many rows
        self.min_impurity_decrease = min_impurity_decrease    # ignore splits that help less than this
        self.ccp_alpha = ccp_alpha                            # pruning strength: bigger = smaller tree

    # ------------------------------------------------------------ Part 1: building
    def _build(self, X, y, depth, available=None):
        node = self._make_node(y)
        if self._stop(node, depth):
            return node
        parent = gini(y)

        best = None  # (impurity decrease, feature, kind, threshold)
        for f in self.feature_names_:
            col = X[f].to_numpy()
            if self.is_numeric_[f]:
                found = self._best_numeric(col.astype(float), y, parent)
                kind = "numeric"
            else:
                found = self._best_categorical(col, y, parent)
                kind = "subset"
            if found and (best is None or found[0] > best[0]):
                best = (found[0], f, kind, found[1])

        if best is None or best[0] < self.min_impurity_decrease:
            return node

        _, feature, kind, threshold = best
        node.feature, node.kind, node.threshold = feature, kind, threshold
        col = X[feature].to_numpy()
        if kind == "numeric":
            left, names = col.astype(float) <= threshold, ("<=", ">")
        else:
            left, names = np.isin(col, list(threshold)), ("in", "out")
        node.children[names[0]] = self._build(X[left], y[left], depth + 1)
        node.children[names[1]] = self._build(X[~left], y[~left], depth + 1)
        return node

    def _best_numeric(self, col, y, parent):
        """Best 'col <= t' split. Returns (impurity decrease, t) or None."""
        n = len(y)
        order = np.argsort(col, kind="mergesort")
        sorted_x, sorted_y = col[order], y[order]
        onehot = np.zeros((n, len(self.classes_)))
        onehot[np.arange(n), np.searchsorted(self.classes_, sorted_y)] = 1
        left_counts = np.cumsum(onehot, axis=0)
        total = left_counts[-1]

        best = None
        for i in range(n - 1):
            if sorted_x[i] == sorted_x[i + 1]:
                continue
            n_left, n_right = i + 1, n - i - 1
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            lc = left_counts[i]
            child = (n_left / n) * gini_from_counts(lc) + (n_right / n) * gini_from_counts(total - lc)
            decrease = parent - child
            if best is None or decrease > best[0]:
                best = (decrease, float((sorted_x[i] + sorted_x[i + 1]) / 2))  # midpoint
        return best

    def _best_categorical(self, col, y, parent):
        """Best 'col in S' split over groupings S of the values. Returns (decrease, S) or None."""
        values = sorted(set(col.tolist()), key=str)
        k, n = len(values), len(y)
        if k < 2:
            return None
        per_value = {v: np.array([np.sum((col == v) & (y == c)) for c in self.classes_])
                     for v in values}
        total = sum(per_value.values())

        if k <= self.MAX_EXHAUSTIVE_VALUES:
            # every way to cut the values into two groups, each counted once:
            # the group WITHOUT values[0] is the "in" group
            groupings = (g for r in range(1, k) for g in combinations(values[1:], r))
        else:
            groupings = ((v,) for v in values)  # too many values: one value vs the rest

        best = None
        for group in groupings:
            lc = sum(per_value[v] for v in group)
            n_left = int(lc.sum())
            n_right = n - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            child = (n_left / n) * gini_from_counts(lc) + (n_right / n) * gini_from_counts(total - lc)
            decrease = parent - child
            if best is None or decrease > best[0]:
                best = (decrease, frozenset(group))
        return best

    # ------------------------------------------------------------ Part 2: pruning
    def _post_fit(self):
        if self.ccp_alpha > 0:
            self._prune_by_cost_complexity()

    def _prune_by_cost_complexity(self):
        """Keep cutting the 'weakest link' until every remaining branch is worth its size.

        For an inner node t with subtree T_t:
            g(t) = (error if t were a leaf - error of T_t) / (leaves of T_t - 1)
        = how much training error each extra leaf buys. Cut the node with the smallest g
        while g <= ccp_alpha."""
        total_rows = self.root_.n_samples
        while not self.root_.is_leaf:
            weakest, smallest_g = None, np.inf
            for node in self._internal_nodes(self.root_):
                leaves = self._leaves(node)
                error_as_leaf = node.errors / total_rows
                error_of_subtree = sum(leaf.errors for leaf in leaves) / total_rows
                g = (error_as_leaf - error_of_subtree) / (len(leaves) - 1)
                if g < smallest_g:
                    weakest, smallest_g = node, g
            if smallest_g > self.ccp_alpha:
                break
            weakest.collapse()
