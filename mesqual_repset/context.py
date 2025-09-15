from __future__ import annotations

import copy
from typing import List, Dict, Hashable, TYPE_CHECKING
from dataclasses import dataclass, field

import pandas as pd

if TYPE_CHECKING:
    from .time_slicer import TimeSlicer


@dataclass
class ProblemContext:
    """A data container passed through the entire workflow."""
    df_raw: pd.DataFrame
    slicer: TimeSlicer
    variable_weights: Dict = field(default_factory=dict)
    feature_weights: Dict = field(default_factory=dict)
    _df_features: pd.DataFrame = None

    def copy(self) -> 'ProblemContext':
        """
        Creates a deep copy of this ProblemContext instance.

        Returns
        -------
        ProblemContext
            A new, independent instance of the context.
        """
        return copy.deepcopy(self)

    def get_sliced_data(self) -> Dict[Hashable, pd.DataFrame]:
        """Generates sliced raw data on demand."""
        raise NotImplementedError

    @property
    def df_features(self) -> pd.DataFrame:
        if self._df_features is None:
            raise ValueError(
                f'You tried to retrieve df_features before assigning it. Please set first using a FeatureEngineer.'
            )
        return self._df_features

    @df_features.setter
    def df_features(self, df_features: pd.DataFrame):
        self._df_features = df_features

    def get_unique_slices(self) -> List[Hashable]:
        return self.slicer.unique_slices(self.df_raw.index)
