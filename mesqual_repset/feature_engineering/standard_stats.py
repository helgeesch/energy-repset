from __future__ import annotations
from typing import Literal, List, Dict, TYPE_CHECKING
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .base_feature_engineer import FeatureEngineer

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..time_slicer import TimeSlicer


class StandardStatsFeatureEngineer(FeatureEngineer):
    """Extracts statistical features from time-series slices with robust scaling.

    For each original variable and slice, computes:
    - Central tendency: mean, median (q50)
    - Dispersion: std, IQR (q90 - q10), q10, q90
    - Distribution shape: neg_share (proportion of negative values)
    - Temporal dynamics: ramp_std (std of first differences)

    Optionally includes cross-variable correlations within each slice (upper
    triangle only, Fisher-z transformed). Features are z-score normalized
    across slices to ensure comparability.

    Examples:
        >>> engineer = StandardStatsFeatureEngineer()
        >>> context_with_features = engineer.run(context)
        >>> print(context_with_features.df_features.columns)
        # ['mean__demand', 'mean__solar', 'std__demand', 'std__solar', ...]

        >>> engineer_no_corr = StandardStatsFeatureEngineer(
        ...     include_correlations=False,
        ...     scale='zscore'
        ... )
        >>> context_with_features = engineer_no_corr.run(context)
        >>> print(context_with_features.df_features.shape)
        # (12, 16) for 12 months, 2 variables, 8 stats each
    """

    def __init__(
            self,
            include_correlations: bool = True,
            scale: Literal["zscore", "none"] = "zscore",
            min_rows_for_corr: int = 8,
    ):
        """Initialize the statistical feature engineer.

        Args:
            include_correlations: If True, include cross-variable correlations
                per slice (Fisher-z transformed).
            scale: Scaling method. Currently only "zscore" is fully supported.
            min_rows_for_corr: Minimum number of rows per slice required to
                compute correlations. Slices with fewer rows get correlation
                features set to 0.
        """
        self.include_correlations = include_correlations
        self.scale = scale
        self.min_rows_for_corr = min_rows_for_corr

        self._raw_feats_: pd.Series = None
        self._means_: pd.Series = None
        self._stds_: pd.Series = None
        self._feature_names_: List[str] = None

    def _calc_and_get_features_df(self, context: "ProblemContext") -> pd.DataFrame:
        """Calculate statistical features and return scaled feature matrix.

        Args:
            context: Problem context with raw time-series data.

        Returns:
            DataFrame where each row is a slice and columns are scaled statistical
            features. Column names follow pattern '{stat}__{variable}'.
        """
        self._fit(context)
        return self._transform(context)

    def _fit(self, context: "ProblemContext") -> None:
        """Compute raw features and fit scaling parameters."""
        df_raw = context.df_raw
        slicer = context.slicer

        self._raw_feats_ = self._compute_raw_features(df_raw, slicer)
        if self.scale == "zscore":
            self._means_ = self._raw_feats_.mean(axis=0)
            self._stds_ = self._raw_feats_.std(axis=0).replace(0, 1.0)
        self._feature_names_ = list(self._raw_feats_.columns)

    def _transform(self, context: "ProblemContext") -> pd.DataFrame:
        """Apply scaling to raw features."""
        feats = self._raw_feats_
        if self.scale == "zscore":
            feats = (feats - self._means_) / self._stds_
        else:
            raise NotImplementedError(f"Scaling {self.scale} not recognized.")
        feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return feats

    def feature_names(self) -> List[str]:
        """Get list of feature column names.

        Returns:
            List of feature names in the format '{stat}__{variable}' or
            'corr__{var1}__{var2}' for correlations.
        """
        if self._feature_names_ is None:
            return []
        return list(self._feature_names_)

    def _compute_raw_features(self, df: pd.DataFrame, slicer: TimeSlicer) -> pd.DataFrame:
        """Compute raw (unscaled) statistical features for each slice."""
        X = df.select_dtypes(include=[np.number]).copy()
        labels = pd.Index(slicer.labels_for_index(X.index), name="slice")
        grp = X.groupby(labels)

        def neg_share(a: pd.Series) -> float:
            n = a.notna().sum()
            return float((a < 0).sum() / n) if n > 0 else 0.0

        def ramp_std(a: pd.Series) -> float:
            d = a.diff().dropna()
            return float(d.std()) if len(d) else 0.0

        stats: Dict[str, pd.DataFrame] = {}
        stats["mean"] = grp.mean(numeric_only=True)
        stats["std"] = grp.std(numeric_only=True).fillna(0.0)
        stats["q10"] = grp.quantile(0.10)
        stats["q50"] = grp.quantile(0.50)
        stats["q90"] = grp.quantile(0.90)
        stats["iqr"] = stats["q90"] - stats["q10"]
        stats["neg_share"] = grp.apply(lambda g: g.apply(neg_share, axis=0))
        stats["ramp_std"] = grp.apply(lambda g: g.apply(ramp_std, axis=0))

        frames = []
        for key, dfk in stats.items():
            dfk = dfk.add_prefix(f"{key}__")
            frames.append(dfk)

        if self.include_correlations and X.shape[1] >= 2:
            cols = list(X.columns)
            pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
            names = [f"corr__{cols[i]}__{cols[j]}" for i, j in pairs]
            corr_rows = []
            idx_rows = []
            for s, g in grp:
                if len(g) >= self.min_rows_for_corr:
                    C = g.corr().to_numpy()
                    vals = [C[i, j] for i, j in pairs]
                else:
                    vals = [0.0] * len(pairs)
                zvals = [0.5 * np.log((1 + v) / (1 - v)) if abs(v) < 0.999 else np.sign(v) * 3.8 for v in vals]
                corr_rows.append(zvals)
                idx_rows.append(s)
            corr_df = pd.DataFrame(corr_rows, index=idx_rows, columns=names)
            frames.append(corr_df)

        df_features = pd.concat(frames, axis=1).sort_index()
        return df_features
