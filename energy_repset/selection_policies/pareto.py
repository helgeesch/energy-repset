from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal, Tuple, Hashable, TYPE_CHECKING

import numpy as np
import pandas as pd

from .policy import PolicyOutcome, SelectionPolicy

if TYPE_CHECKING:
    from ..objectives import ObjectiveSet
    from ..types import ScoreComponentDirection


Normalization = Literal["none", "robust_minmax", "zscore_iqr"]


@dataclass(frozen=True)
class ParetoOutcome(PolicyOutcome):
    objectives: Dict[str, ScoreComponentDirection]
    feasible_mask_col: str
    pareto_mask_col: str
    score_col: str


class ParetoUtopiaPolicy(SelectionPolicy):
    def __init__(
            self,
            objectives: Optional[Dict[str, ScoreComponentDirection]] = None,
            normalization: Normalization = "robust_minmax",
            fairness_constraints: Optional[Dict[str, float]] = None,
            distance: Literal["chebyshev", "euclidean"] = "chebyshev",
            tie_breakers: Tuple[str, ...] = (),
            tie_dirs: Tuple[ScoreComponentDirection, ...] = (),
            eps: float = 1e-9,
    ) -> None:
        self.objectives = objectives
        self.normalization = normalization
        self.fairness_constraints = fairness_constraints or {}
        self.distance = distance
        self.tie_breakers = tie_breakers
        self.tie_dirs = tie_dirs
        self.eps = eps
        self.pareto_mask: pd.Series | None = None
        self.feasible_mask: pd.Series | None = None

    def select_best(self, evaluations_df: pd.DataFrame, objective_set: ObjectiveSet) -> Tuple[Hashable, ...]:
        """Select best solution using Pareto utopia approach."""
        df = evaluations_df.copy()
        dirs = self._resolve_objectives_from_meta(objective_set.component_meta(), df)
        feas = self._apply_constraints(df, self.fairness_constraints)
        df["feasible"] = feas
        Y = df[list(dirs.keys())].copy()

        # Orient all objectives for minimization
        for c, d in dirs.items():
            if d == "max":
                Y[c] = -Y[c]

        # Find Pareto front
        pareto_mask = self._pareto_mask(Y[feas])
        df["pareto"] = False
        df.loc[feas.index[feas].tolist(), "pareto"] = pareto_mask.values

        # Store masks for diagnostics
        self.pareto_mask = df["pareto"].copy()
        self.feasible_mask = df["feasible"].copy()

        # Normalize and compute utopia distance
        Z = self._normalize(Y, self.normalization)
        ideal = Z[df["feasible"]].min(axis=0)
        dist = self._dist(Z, ideal, self.distance)
        df["utopia_distance"] = dist

        # Select from Pareto front
        front = df[(df["feasible"]) & (df["pareto"])]
        if len(front) == 0:
            front = df[df["feasible"]] if df["feasible"].any() else df

        best = front.sort_values("utopia_distance", ascending=True)
        if len(best) > 1 and len(self.tie_breakers) > 0:
            for col, d in zip(self.tie_breakers, self.tie_dirs):
                best = best.sort_values(col, ascending=(d == "min"))

        return tuple(best.iloc[0]["slices"])

    def _resolve_objectives(self, objective_set: ObjectiveSet, df: pd.DataFrame) -> Dict[str, ScoreComponentDirection]:
        """Legacy method for backward compatibility."""
        if self.objectives is not None:
            return self.objectives
        meta = objective_set.component_meta()
        return {name: info["direction"] for name, info in meta.items() if name in df.columns}

    def _resolve_objectives_from_meta(self, meta: Dict[str, Dict[str, any]], df: pd.DataFrame) -> Dict[str, ScoreComponentDirection]:
        """Resolve objectives from component metadata."""
        if self.objectives is not None:
            return self.objectives
        return {name: info["direction"] for name, info in meta.items() if name in df.columns}

    def _apply_constraints(self, df: pd.DataFrame, cons: Dict[str, float]) -> pd.Series:
        if not cons:
            return pd.Series(True, index=df.index)
        mask = pd.Series(True, index=df.index)
        for col, thr in cons.items():
            if col not in df.columns:
                raise ValueError(f"Unknown constraint metric: {col}")
            mask &= df[col] <= thr
        return mask

    def _normalize(self, Y: pd.DataFrame, mode: Normalization) -> pd.DataFrame:
        if mode == "robust_minmax":
            q_lo = Y.quantile(0.05)
            q_hi = Y.quantile(0.95)
            denom = (q_hi - q_lo).replace(0, 1.0)
            return ((Y - q_lo) / denom).clip(lower=0.0)
        if mode == "zscore_iqr":
            med = Y.median()
            iqr = (Y.quantile(0.75) - Y.quantile(0.25)).replace(0, 1.0)
            return (Y - med) / iqr
        return Y

    def _dist(self, Z: pd.DataFrame, ideal: pd.Series, kind: str) -> pd.Series:
        D = (Z - ideal.values)
        if kind == "chebyshev":
            return D.abs().max(axis=1)
        return np.sqrt((D.pow(2)).sum(axis=1))

    def _pareto_mask(self, Y: pd.DataFrame) -> pd.Series:
        A = Y.values
        n = A.shape[0]
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(A[j], A[i]):
                    mask[i] = False
                    break
        return pd.Series(mask, index=Y.index)

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a <= b + self.eps) and np.any(a < b - self.eps)


class ParetoMaxMinStrategy(ParetoUtopiaPolicy):

    def select_best(self, evaluations_df: pd.DataFrame, objective_set: ObjectiveSet) -> Tuple[Hashable, ...]:
        """Select best solution using Pareto max-min approach."""
        df = evaluations_df.copy()
        dirs = self._resolve_objectives_from_meta(objective_set.component_meta(), df)
        feas = self._apply_constraints(df, self.fairness_constraints)
        df["feasible"] = feas
        Y = df[list(dirs.keys())].copy()

        # Orient all objectives for minimization
        for c, d in dirs.items():
            if d == "max":
                Y[c] = -Y[c]

        # Find Pareto front
        pareto_mask = self._pareto_mask(Y[feas])
        df["pareto"] = False
        df.loc[feas.index[feas].tolist(), "pareto"] = pareto_mask.values

        # Store masks for diagnostics
        self.pareto_mask = df["pareto"].copy()
        self.feasible_mask = df["feasible"].copy()

        # Normalize and compute max-min score
        Z = self._normalize(Y, self.normalization)
        ideal = Z[df["feasible"]].min(axis=0)
        slack = 1.0 - (Z - ideal.values)
        df["maxmin_score"] = slack.min(axis=1)

        # Select from Pareto front
        front = df[(df["feasible"]) & (df["pareto"])]
        if len(front) == 0:
            front = df[df["feasible"]] if df["feasible"].any() else df

        best = front.sort_values("maxmin_score", ascending=False)
        if len(best) > 1 and len(self.tie_breakers) > 0:
            for col, d in zip(self.tie_breakers, self.tie_dirs):
                best = best.sort_values(col, ascending=(d == "min"))

        return tuple(best.iloc[0]["slices"])
