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
    """
    Default slice-level features with robust scaling.

    Features per original variable v and slice s:
      - mean, std
      - q10, q50, q90, iqr
      - neg_share (share of values < 0)
      - ramp_std (std of first differences)
    Plus cross-variable correlations within slice (upper triangle, Fisher-z).

    Options
    -------
    include_correlations : include cross-variable correlations per slice
    scale : "zscore" to z-score features across slices; "none" to skip scaling
    min_rows_for_corr : min rows per slice to compute correlations
    """

    def __init__(
            self,
            include_correlations: bool = True,
            scale: Literal["zscore", "none"] = "zscore",
            min_rows_for_corr: int = 8,
    ):
        self.include_correlations = include_correlations
        self.scale = scale
        self.min_rows_for_corr = min_rows_for_corr

        self._raw_feats_: pd.Series = None
        self._means_: pd.Series = None
        self._stds_: pd.Series = None
        self._feature_names_: List[str] = None

    def _calc_and_get_features_df(self, context: "ProblemContext") -> pd.DataFrame:
        self._fit(context)
        return self._transform(context)

    def _fit(self, context: "ProblemContext") -> None:
        df_raw = context.df_raw
        slicer = context.slicer

        self._raw_feats_ = self._compute_raw_features(df_raw, slicer)
        if self.scale == "zscore":
            self._means_ = self._raw_feats_.mean(axis=0)
            self._stds_ = self._raw_feats_.std(axis=0).replace(0, 1.0)
        self._feature_names_ = list(self._raw_feats_.columns)

    def _transform(self, context: "ProblemContext") -> pd.DataFrame:
        feats = self._raw_feats_
        if self.scale == "zscore":
            feats = (feats - self._means_) / self._stds_
        else:
            raise NotImplementedError(f"Scaling {self.scale} not recognized.")
        feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return feats

    def feature_names(self) -> List[str]:
        if self._feature_names_ is None:
            return []
        return list(self._feature_names_)

    def _compute_raw_features(self, df: pd.DataFrame, slicer: TimeSlicer) -> pd.DataFrame:
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
