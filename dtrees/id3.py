"""ID3 (Quinlan, 1986): categorical features, information gain, multiway splits."""
import numpy as np

from .base import BaseDecisionTree, entropy, partition_masks


class ID3DecisionTree(BaseDecisionTree):
    def _validate_features(self):
        numeric = [c for c, is_num in self.is_numeric_.items() if is_num]
        if numeric:
            raise ValueError(
                f"ID3 only supports categorical features, but {numeric} are numeric. "
                "Turn them into categories first, or use C4.5 / CART."
            )

    @staticmethod
    def information_gain(col, y):
        """Gain = entropy(parent) - weighted average entropy of the child groups."""
        n = len(y)
        remainder = sum(
            (mask.sum() / n) * entropy(y[mask]) for mask in partition_masks(col).values()
        )
        return entropy(y) - remainder

    def _build(self, X, y, depth, available):
        node = self._make_node(y)
        if self._stop(node, depth) or not available:
            return node

        gains = {f: self.information_gain(X[f].to_numpy(), y) for f in available}
        best = max(gains, key=gains.get)  # on a tie, the first feature wins

        node.feature, node.kind = best, "multiway"
        remaining = [f for f in available if f != best]  # ID3 never reuses a feature on a path
        col = X[best].to_numpy()
        for value, mask in partition_masks(col).items():
            node.children[value] = self._build(X[mask], y[mask], depth + 1, remaining)
        return node
