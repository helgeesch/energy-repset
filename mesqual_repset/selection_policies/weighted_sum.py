from __future__ import annotations

from typing import Optional, Dict, Tuple, Hashable, Literal

import pandas as pd

from ..objectives import ObjectiveSet
from ..types import ScoreComponentDirection
from .policy import SelectionPolicy

Normalization = Literal["none", "robust_minmax", "zscore_iqr"]


class WeightedSumPolicy(SelectionPolicy):
    def __init__(
            self,
            overrides: Optional[Dict[str, float]] = None,
            normalization: Normalization = "none",
            tie_breakers: Tuple[str, ...] = (),
            tie_dirs: Tuple[ScoreComponentDirection, ...] = (),
    ) -> None:
        self.overrides = overrides or {}
        self.normalization = normalization
        self.tie_breakers = tie_breakers
        self.tie_dirs = tie_dirs

    def select_best(self, evaluations_df: pd.DataFrame, objective_set: ObjectiveSet) -> Tuple[Hashable, ...]:
        """Select best solution using weighted sum approach."""
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
