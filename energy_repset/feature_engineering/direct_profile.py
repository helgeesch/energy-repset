from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd

from .base_feature_engineer import FeatureEngineer

if TYPE_CHECKING:
    from ..context import ProblemContext


class DirectProfileFeatureEngineer(FeatureEngineer):
    """Feature engineer that uses raw profile vectors directly (F_direct).

    For each slice, concatenates the raw hourly values across all variables
    into a single flat feature vector. This preserves the full temporal shape
    of each period, making it suitable for algorithms that compare time-series
    profiles directly (e.g., Snippet Algorithm, DTW-based methods).

    Args:
        variable_weights: Optional dict mapping column names to scalar weights.
            Weighted columns are multiplied by their weight before flattening.
            Columns not in the dict are included with weight 1.0.

    Examples:
        Basic usage with daily slicing:

        >>> from energy_repset.feature_engineering import DirectProfileFeatureEngineer
        >>> engineer = DirectProfileFeatureEngineer()
        >>> context_with_features = engineer.run(context)
        >>> context_with_features.df_features.shape
        (365, 72)  # 365 days x (24 hours * 3 variables)
    """

    def __init__(self, variable_weights: Optional[Dict[str, float]] = None):
        """Initialize direct profile feature engineer.

        Args:
            variable_weights: Optional mapping of variable names to weights.
                Variables not in the dict receive weight 1.0.
        """
        self.variable_weights = variable_weights or {}

    def calc_and_get_features_df(self, context: ProblemContext) -> pd.DataFrame:
        """Flatten each slice's raw values into a single feature row.

        Args:
            context: Problem context with raw time-series data.

        Returns:
            DataFrame where each row is one slice and columns are the
            flattened hourly values (hours x variables).
        """
        df = context.df_raw.select_dtypes(include=[np.number]).copy()

        for col, w in self.variable_weights.items():
            if col in df.columns:
                df[col] = df[col] * w

        slice_labels = context.slicer.labels_for_index(df.index)
        unique_slices = context.slicer.unique_slices(df.index)

        rows = []
        for s in unique_slices:
            mask = slice_labels == s
            chunk = df.loc[mask]
            flat = chunk.values.flatten(order='C')
            rows.append(flat)

        max_len = max(len(r) for r in rows)
        padded = []
        for r in rows:
            if len(r) < max_len:
                r = np.pad(r, (0, max_len - len(r)), constant_values=np.nan)
            padded.append(r)

        col_names = [f"t{i}" for i in range(max_len)]
        return pd.DataFrame(padded, index=unique_slices, columns=col_names)
