from __future__ import annotations

from typing import Optional, Dict, Tuple, Hashable, Literal

import pandas as pd

from ..objectives import ObjectiveSet
from ..types import ScoreComponentDirection
from .policy import SelectionPolicy

Normalization = Literal["none", "robust_minmax", "zscore_iqr"]


class WeightedSumPolicy(SelectionPolicy):
    """Selects the combination minimizing a weighted sum of objectives.

    Combines multiple objectives into a single scalar score using weighted
    averaging. Objectives are oriented for minimization (max objectives are
    negated), optionally normalized, then combined using weights from the
    ObjectiveSet (which can be overridden).

    This is the simplest multi-objective selection strategy and works well
    when relative importance of objectives is known.

    Examples:
        >>> from energy_repset import ObjectiveSet, ObjectiveSpec
        >>> from energy_repset.score_components import WassersteinFidelity, CorrelationFidelity
        >>> # Default: use weights from ObjectiveSet
        >>> policy = WeightedSumPolicy()
        >>> objectives = ObjectiveSet([
        ...     ObjectiveSpec('wasserstein', WassersteinFidelity(), weight=1.0),
        ...     ObjectiveSpec('correlation', CorrelationFidelity(), weight=0.5)
        ... ])
        >>> # Final score = 1.0*wasserstein + 0.5*correlation

        >>> # Override weights in policy
        >>> policy = WeightedSumPolicy(
        ...     overrides={'wasserstein': 2.0, 'correlation': 1.0}
        ... )
        >>> # Final score = 2.0*wasserstein + 1.0*correlation

        >>> # With normalization to make objectives comparable
        >>> policy = WeightedSumPolicy(
        ...     normalization='robust_minmax',  # Scale to [0, 1] using 5th-95th percentiles
        ...     tie_breakers=('wasserstein',),  # Break ties by wasserstein
        ...     tie_dirs=('min',)
        ... )
    """
    def __init__(
            self,
            overrides: Optional[Dict[str, float]] = None,
            normalization: Normalization = "none",
            tie_breakers: Tuple[str, ...] = (),
            tie_dirs: Tuple[ScoreComponentDirection, ...] = (),
    ) -> None:
        """Initialize weighted sum policy.

        Args:
            overrides: Optional dict mapping objective names to weights,
                overriding weights from ObjectiveSet.
            normalization: How to normalize objectives before weighting:
                - "none": No normalization
                - "robust_minmax": Scale to [0, 1] using 5th-95th percentiles
                - "zscore_iqr": Z-score using median and IQR
            tie_breakers: Tuple of objective names to use for tie-breaking.
            tie_dirs: Corresponding directions ("min" or "max") for tie-breakers.
        """
        self.overrides = overrides or {}
        self.normalization = normalization
        self.tie_breakers = tie_breakers
        self.tie_dirs = tie_dirs

    def select_best(self, evaluations_df: pd.DataFrame, objective_set: ObjectiveSet) -> Tuple[Hashable, ...]:
        """Select combination with minimum weighted sum score.

        Args:
            evaluations_df: DataFrame with 'slices' column and objective scores.
            objective_set: Provides component metadata (direction, weights).

        Returns:
            Tuple of slice identifiers with the lowest weighted sum score.
        """
        df = evaluations_df.copy()
        meta = objective_set.component_meta()
        oriented = df[list(meta.keys())].copy()

        # Orient all objectives for minimization
        for name, m in meta.items():
            if m["direction"] == "max":
                oriented[name] = -oriented[name]

        # Normalize if requested
        Z = self._normalize(oriented, mode=self.normalization)

        # Compute weights (preferences from ObjectiveSet, overrides from strategy)
        weights = {name: float(m["pref"]) for name, m in meta.items()}
        for k, v in self.overrides.items():
            if k not in weights:
                raise ValueError(f"Unknown metric in overrides: {k}")
            weights[k] = float(v)

        # Compute weighted sum scores
        df["strategy_score"] = sum(Z[name] * w for name, w in weights.items())

        # Find best solution
        best = df.sort_values("strategy_score", ascending=True)
        if len(best) > 1 and len(self.tie_breakers) > 0:
            for col, d in zip(self.tie_breakers, self.tie_dirs):
                best = best.sort_values(col, ascending=(d == "min"))

        return tuple(best.iloc[0]["slices"])

    def _normalize(self, Y: pd.DataFrame, mode: Normalization) -> pd.DataFrame:
        if mode == "none":
            return Y
        if mode == "robust_minmax":
            q_lo = Y.quantile(0.05)
            q_hi = Y.quantile(0.95)
            denom = (q_hi - q_lo).replace(0, 1.0)
            return ((Y - q_lo) / denom).clip(lower=0.0)
        med = Y.median()
        iqr = (Y.quantile(0.75) - Y.quantile(0.25)).replace(0, 1.0)
        return (Y - med) / iqr
