from __future__ import annotations
from typing import TYPE_CHECKING, Dict
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..types import SliceCombination
    from .base_score_component import ScoreComponent
    from ..context import ProblemContext


class DiurnalFidelity(ScoreComponent):
    """
    Preserves intraday shape by comparing mean profiles over hour-of-day.
    """

    def __init__(self) -> None:
        self.name = "diurnal"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """
        Precompute the full dataset's diurnal mean profile.

        Parameters
        ----------
        context
            Your problems context.
        """
        df = context.df_raw.copy()
        slicer = context.slicer
        self.df = df
        self.labels = slicer.labels_for_index(df.index)
        self.full = df.groupby(df.index.hour).mean(numeric_only=True)

    def score(self, combination: SliceCombination) -> float:
        """
        Compute the relative MSE between full and selection diurnal profiles.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Normalized MSE across variables and hours.
        """
        sel = self.df.loc[pd.Index(self.labels).isin(selection)]
        sub = sel.groupby(sel.index.hour).mean(numeric_only=True)
        a, b = self.full.align(sub, join="inner", axis=0)
        num = float(((a - b).pow(2)).mean().mean())
        den = float(a.pow(2).mean().mean()) + 1e-12
        return num / den


class DiversityReward(ScoreComponent):
    """
    Rewards selections whose slice features are mutually distant on average.
    """

    def __init__(self) -> None:
        self.name = "diversity"
        self.direction = "max"

    def prepare(self, context: ProblemContext) -> None:
        """
        Store standardized per-slice features.

        Parameters
        ----------
        context
            Your problems context.
        """
        self.features = context.df_features.copy()

    def score(self, combination: SliceCombination) -> float:
        """
        Compute mean pairwise Euclidean distance among selected slice features.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Average pairwise distance; zero if fewer than two slices.
        """
        X = self.features.loc[list(combination)].values
        if X.shape[0] < 2:
            return 0.0
        n = X.shape[0]
        dsum = 0.0
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                dsum += float(np.linalg.norm(X[i] - X[j]))
                cnt += 1
        return dsum / cnt


class CentroidBalance(ScoreComponent):
    """
    Keeps the centroid of selected slice features close to the global center.
    """

    def __init__(self) -> None:
        self.name = "centroid_balance"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """
        Store standardized per-slice features.

        Parameters
        ----------
        context
            Your problems context.
        """
        self.features = context.df_features.copy()

    def score(self, combination: SliceCombination) -> float:
        """
        Compute Euclidean distance of the selection centroid to the origin.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Centroid distance.
        """
        X = self.features.loc[list(combination)].values
        mu = X.mean(axis=0)
        return float(np.linalg.norm(mu))


class CoverageBalance(ScoreComponent):
    """
    Promotes balanced coverage by encouraging uniform responsibility across selected slices.

    Parameters
    ----------
    gamma
        RBF kernel sharpness used to compute soft assignments.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        self.name = "coverage_balance"
        self.direction = "min"
        self.gamma = gamma

    def prepare(self, context: ProblemContext) -> None:
        """
        Precompute feature matrices used for responsibilities.

        Parameters
        ----------
        context
            Your problems context.
        """
        self.features = context.df_features.copy()
        self.all_X = np.nan_to_num(self.features.values, nan=0.0)

    def _responsibilities(self, combination: SliceCombination) -> np.ndarray:
        sel_X = self.features.loc[list(combination)].values
        d2 = ((self.all_X[:, None, :] - sel_X[None, :, :]) ** 2).sum(axis=2)
        K = np.exp(-self.gamma * d2)
        mass = K.sum(axis=0)
        if mass.sum() <= 0:
            return np.ones(len(combination)) / len(combination)
        return mass / mass.sum()

    def score(self, combination: SliceCombination) -> float:
        """
        Compute L2 deviation of responsibility shares from uniform.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Non-negative penalty; zero when perfectly balanced.
        """
        r = self._responsibilities(combination)
        u = np.ones_like(r) / len(r)
        return float(np.linalg.norm(r - u))


class NRMSEFidelity(ScoreComponent):
    """
    Matches per-variable duration curves between full data and selected subset.

    The score is the sum of normalized root-mean-square errors (NRMSE)
    between the duration curve of the full dataset and an interpolated
    duration curve of the selection. This provides a measure of how well
    the selection preserves the statistical distribution of each variable.

    Parameters
    ----------
    variable_weights
        Optional per-variable weights applied inside the component.
    """

    def __init__(self, variable_weights: Dict[str, float] | None = None) -> None:
        self.name = "nrmse"
        self.direction = "min"
        self.variable_weights = variable_weights

    def prepare(self, context: ProblemContext) -> None:
        """
        Precompute full duration curves and means for normalization.

        Parameters
        ----------
        df
            Input time-series DataFrame.
        slicer
            TimeSlicer to align selections.
        features
            Per-slice feature matrix, unused here but required by interface.
        """
        df = context.df_raw
        self.df = df
        self.labels = context.slicer.labels_for_index(df.index)
        self.vars = list(df.columns)

        if self.variable_weights is None:
            self.variable_weights = {v: 1.0 for v in self.vars}
        for v in self.vars:
            if v not in self.variable_weights:
                self.variable_weights[v] = 1.0

        self.full_curves = {
            v: np.sort(df[v].values)[::-1] for v in self.vars
        }
        self.full_means = {
            v: np.mean(df[v].values) for v in self.vars
        }

    def score(self, combination: SliceCombination) -> float:
        """
        Compute the NRMSE fidelity score based on duration curves.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Sum of per-variable NRMSE of the duration curves.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        if not sel_mask.any():
            return np.inf

        sel = self.df.loc[sel_mask]
        s = 0.0

        for v in self.vars:
            full_curve = self.full_curves[v]
            sel_curve = np.sort(sel[v].values)[::-1]

            if len(sel_curve) == 0:
                continue

            # Interpolate selection's duration curve to match the length of the full curve
            x_full = np.linspace(0, 1, len(full_curve))
            x_sel = np.linspace(0, 1, len(sel_curve))
            resampled_sel_curve = np.interp(x_full, x_sel, sel_curve)

            # Calculate RMSE
            mse = np.mean((full_curve - resampled_sel_curve) ** 2)
            rmse = np.sqrt(mse)

            # Normalize by the mean of the full data
            mean_val = self.full_means[v]
            nrmse = rmse / (mean_val + 1e-12)

            s += self.variable_weights[v] * nrmse

        return float(s)


class DTWFidelity(ScoreComponent):
    """
    Measures how well the selected slices represent the entire dataset
    using the Dynamic Time Warping (DTW) distance.

    The score is the average DTW distance between each unselected slice
    and its closest representative slice in the selection. A lower score
    indicates a better representation. This is analogous to the inertia
    in k-medoids clustering.
    """

    def __init__(self) -> None:
        self.name = "dtw"
        self.direction = "min"

    def prepare(self, context: ProblemContext) -> None:
        """
        Precompute per-slice time series data.

        Parameters
        ----------
        df
            Input time-series DataFrame.
        slicer
            TimeSlicer to align selections.
        features
            Per-slice feature matrix, unused here but required by interface.
        """
        df = context.df_raw
        self.slices = {
            label: group.values
            for label, group in df.groupby(context.slicer.labels_for_index(df.index))
        }
        self.all_labels = set(self.slices.keys())

    def score(self, combination: SliceCombination) -> float:
        """
        Compute the DTW fidelity score.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Average DTW distance from unselected slices to the selection.
        """
        from tslearn.metrics import dtw
        selected_labels = set(combination)
        unselected_labels = self.all_labels - selected_labels

        if not selected_labels or not unselected_labels:
            return 0.0

        selected_series = [self.slices[lbl] for lbl in selected_labels]
        total_dist = 0.0

        for lbl in unselected_labels:
            unselected_series = self.slices[lbl]
            min_dist = np.inf
            for sel_series in selected_series:
                dist = dtw(unselected_series, sel_series)
                if dist < min_dist:
                    min_dist = dist
            total_dist += min_dist

        return total_dist / len(unselected_labels)


class DurationCurveFidelity(ScoreComponent):
    # TODO: Might be duplicate of NRMSE from above? Just pasted blindly
    """
    Matches per-variable duration curves between full data and selected subset.
    The duration curve is approximated by a set of quantiles. The score
    is the normalized root mean square error (NRMSE) between the quantiles.

    Parameters
    ----------
    n_quantiles
        Number of quantiles to use for approximating the duration curve.
    variable_weights
        Optional per-variable weights applied inside the component.
    """

    def __init__(self, n_quantiles: int = 101, variable_weights: Dict[str, float] | None = None) -> None:
        self.name = "nrmse_duration_curve"
        self.direction = "min"
        self.n_quantiles = n_quantiles
        self.variable_weights = variable_weights

    def prepare(self, context: ProblemContext) -> None:
        """
        Precompute quantiles for the full dataset.

        Parameters
        ----------
        df
            Input time-series DataFrame.
        slicer
            TimeSlicer to align selections.
        features
            Per-slice feature matrix, unused here but required by interface.
        """
        df = context.df_raw
        self.df = df
        self.labels = context.slicer.labels_for_index(df.index)
        self.vars = list(df.columns)

        if self.variable_weights is None:
            self.variable_weights = {v: 1.0 for v in self.vars}
        for v in self.vars:
            if v not in self.variable_weights:
                self.variable_weights[v] = 1.0

        self.quantiles = np.linspace(0, 1, self.n_quantiles)
        self.full_quantiles = self.df.quantile(self.quantiles)
        self.iqr = (df.quantile(0.75) - df.quantile(0.25)).replace(0, 1.0)

    def score(self, combination: SliceCombination) -> float:
        """
        Compute the NRMSE score for the duration curves.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Sum of per-variable NRMSE scores.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        if not sel_mask.any():
            return np.inf
        sel = self.df.loc[sel_mask]

        sel_quantiles = sel.quantile(self.quantiles)

        total_nrmse = 0.0
        for v in self.vars:
            squared_errors = (self.full_quantiles[v].values - sel_quantiles[v].values) ** 2
            rmse = np.sqrt(squared_errors.mean())
            total_nrmse += self.variable_weights[v] * (rmse / float(self.iqr[v]))

        return float(total_nrmse)


class DiurnalDTWFidelity(ScoreComponent):
    # TODO: Might be duplicate of DTW from above? Just pasted blindly
    """
    Preserves intraday shape by comparing mean profiles over hour-of-day
    using Dynamic Time Warping (DTW) distance.
    """

    def __init__(self) -> None:
        self.name = "diurnal_dtw"
        self.direction = "min"

    def _dtw_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Computes Dynamic Time Warping distance between two 1D arrays."""
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s1[i - 1] - s2[j - 1])
                last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix[n, m]

    def prepare(self, context: ProblemContext) -> None:
        """
        Precompute the full dataset's diurnal mean profile and normalization factors.

        Parameters
        ----------
        df
            Input time-series DataFrame.
        slicer
            TimeSlicer to align selections.
        features
            Per-slice feature matrix, unused here but required by interface.
        """
        df = context.df_raw
        self.df = df
        self.labels = context.slicer.labels_for_index(df.index)
        self.vars = list(df.columns)
        self.full_diurnal = df.groupby(df.index.hour).mean(numeric_only=True)
        self.norm_factor = self.full_diurnal.std().replace(0, 1.0)

    def score(self, combination: SliceCombination) -> float:
        """
        Compute the normalized DTW distance between full and selection diurnal profiles.

        Parameters
        ----------
        combination
            Slice labels forming the selection.

        Returns
        -------
        float
            Sum of per-variable normalized DTW distances.
        """
        sel_mask = pd.Index(self.labels).isin(combination)
        if not sel_mask.any():
            return np.inf

        sel = self.df.loc[sel_mask]
        sel_diurnal = sel.groupby(sel.index.hour).mean(numeric_only=True)

        full_aligned, sel_aligned = self.full_diurnal.align(sel_diurnal, join="inner", axis=0)
        if full_aligned.empty:
            return np.inf

        total_dtw_dist = 0.0
        for v in self.vars:
            if v in full_aligned and v in sel_aligned:
                full_profile = full_aligned[v].values
                sel_profile = sel_aligned[v].values
                dist = self._dtw_distance(full_profile, sel_profile)
                total_dtw_dist += dist / float(self.norm_factor[v])

        return float(total_dtw_dist)
