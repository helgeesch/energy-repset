from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class CorrelationFidelity(ScoreComponent):
    """Preserves cross-variable correlation structure using Frobenius norm.

    Measures how well the selection preserves the correlation structure between
    variables by comparing the full dataset's correlation matrix with the
    selection's correlation matrix. Uses relative Frobenius norm of the
    difference matrix.

    Lower scores indicate better preservation of variable relationships. This
    component is important for downstream modeling tasks that depend on
    realistic co-occurrence patterns (e.g., solar and wind generation).

    Examples:
        >>> component = CorrelationFidelity()
        >>> component.prepare(context)
        >>> score = component.score((0, 3, 6, 9))
        >>> print(f"Correlation mismatch: {score:.3f}")
        # 0.0 would be perfect preservation, 1.0+ indicates poor preservation

        >>> # Combine with Wasserstein in an ObjectiveSet
        >>> from energy_repset import ObjectiveSet, ObjectiveSpec
        >>> objectives = ObjectiveSet([
        ...     ObjectiveSpec('wasserstein', WassersteinFidelity(), weight=1.0),
        ...     ObjectiveSpec('correlation', CorrelationFidelity(), weight=1.0)
        ... ])
    """

    def __init__(self) -> None:
        """Initialize correlation fidelity component."""
        self.name = "correlation"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """Precompute full dataset's correlation matrix.

        Args:
            context: Problem context with raw time-series data.
        """
        df = context.df_raw.copy()
        slicer = context.slicer
        self.df = df
        self.labels = slicer.labels_for_index(df.index)
        self.full_corr = df.corr()


    def score(self, combination: SliceCombination) -> float:
        """Compute relative Frobenius norm of correlation matrix difference.

        Args:
            combination: Slice identifiers forming the selection.

        Returns:
            Relative Frobenius norm ||C_full - C_sel||_F / ||C_full||_F where
            C denotes correlation matrices. Lower is better (0 = perfect match).
        """
        sel = self.df.loc[pd.Index(self.labels).isin(combination)]
        diff = self.full_corr - sel.corr()
        num = float(np.linalg.norm(diff.values, ord="fro"))
        den = float(np.linalg.norm(self.full_corr.values, ord="fro")) + 1e-12
        return num / den
