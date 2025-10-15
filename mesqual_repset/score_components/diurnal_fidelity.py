from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .base_score_component import ScoreComponent

if TYPE_CHECKING:
    from ..types import SliceCombination
    from ..context import ProblemContext


class DiurnalFidelity(ScoreComponent):
    """Measures how well the selection preserves hourly (diurnal) patterns.

    Compares the mean hourly profiles between the full dataset and the
    selected subset. This is useful for applications where intraday patterns
    matter (e.g., electricity demand profiles, solar generation curves).

    The score is the normalized mean squared error between the full and
    selected hour-of-day profiles, averaged across all variables and hours.

    Examples:
        >>> from mesqual_repset.score_components import DiurnalFidelity
        >>> from mesqual_repset.objectives import ObjectiveSet
        >>>
        >>> # Add diurnal fidelity to your objective set
        >>> objectives = ObjectiveSet({
        ...     'diurnal': (1.0, DiurnalFidelity())
        ... })
        >>>
        >>> # For hourly data, this ensures selected periods
        >>> # preserve the typical daily load shape
    """

    def __init__(self) -> None:
        """Initialize diurnal fidelity component."""
        self.name = "diurnal"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """Precompute the full dataset's mean hourly profile.

        Args:
            context: Problem context containing raw time-series data.
        """
        df = context.df_raw.copy()
        slicer = context.slicer
        self.df = df
        self.labels = slicer.labels_for_index(df.index)
        self.full = df.groupby(df.index.hour).mean(numeric_only=True)

    def score(self, combination: SliceCombination) -> float:
        """Compute normalized MSE between full and selection diurnal profiles.

        Args:
            combination: Tuple of slice identifiers forming the selection.

        Returns:
            Normalized mean squared error across all variables and hours.
            Lower values indicate better preservation of diurnal patterns.
        """
        sel = self.df.loc[pd.Index(self.labels).isin(combination)]
        sub = sel.groupby(sel.index.hour).mean(numeric_only=True)
        a, b = self.full.align(sub, join="inner", axis=0)
        num = float(((a - b).pow(2)).mean().mean())
        den = float(a.pow(2).mean().mean()) + 1e-12
        return num / den
