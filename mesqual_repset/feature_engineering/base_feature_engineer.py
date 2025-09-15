from __future__ import annotations

from typing import List, Protocol, TYPE_CHECKING
from abc import ABC, abstractmethod
import pandas as pd

if TYPE_CHECKING:
    from ..context import ProblemContext


class FeatureEngineer(ABC):
    """
    Pillar F: Transforms raw sliced data into a feature matrix (df_features) and generates a context with df_features.
    """
    def run(self, context: ProblemContext) -> ProblemContext:
        """This method calculates features and returns a NEW ProblemContext instance that now contains df_features."""
        context_with_features = context.copy()
        context_with_features.df_features = self._calc_and_get_features_df(context)
        return context_with_features

    @abstractmethod
    def _calc_and_get_features_df(self, context: ProblemContext) -> pd.DataFrame:
        """Calculates features and returns df_features."""
        ...


class FeaturePipeline(FeatureEngineer):
    def __init__(self, engineers: List[FeatureEngineer]):
        self.engineers = engineers

    def _calc_and_get_features_df(self, context: ProblemContext) -> None:
        all_features = []
        for engineer in self.engineers:
            # Each engineer works on a fresh copy of the context
            # to avoid side-effects, and returns its feature df.
            features = engineer._calc_and_get_features_df(context)
            all_features.append(features)

        # Concatenate all generated features
        context.df_features = pd.concat(all_features, axis=1)
