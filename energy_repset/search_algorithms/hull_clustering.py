from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize, nnls

from .search_algorithm import SearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..context import ProblemContext


class HullClusteringSearch(SearchAlgorithm):
    """Greedy forward selection minimizing total projection error (Hull Clustering).

    Implements the constructive hull clustering algorithm from Bahl et al. (2025).
    At each iteration, the candidate that most reduces the total projection error
    is added to the selection set. Each non-selected period is represented as a
    convex or conic combination of the selected hull vertices.

    The algorithm leaves ``weights=None`` in the result so that an external
    ``RepresentationModel`` (typically ``BlendedRepresentationModel``) can compute
    the final soft-assignment weights.

    Args:
        k: Number of representative periods to select.
        hull_type: Type of projection constraint. ``'convex'`` enforces non-negative
            weights that sum to 1. ``'conic'`` enforces only non-negativity.

    Examples:
        Basic usage with blended representation:

        >>> from energy_repset.search_algorithms import HullClusteringSearch
        >>> from energy_repset.representation import BlendedRepresentationModel
        >>> search = HullClusteringSearch(k=4, hull_type='convex')
        >>> repr_model = BlendedRepresentationModel(blend_type='convex')
    """

    def __init__(self, k: int, hull_type: Literal['convex', 'conic'] = 'convex'):
        """Initialize Hull Clustering search.

        Args:
            k: Number of hull vertices (representative periods) to select.
            hull_type: Projection type. ``'convex'`` requires weights >= 0 and
                sum(weights) == 1. ``'conic'`` requires only weights >= 0.
        """
        self.k = k
        self.hull_type = hull_type

    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Find k hull vertices via greedy forward selection.

        Args:
            context: Problem context with ``df_features`` populated.

        Returns:
            RepSetResult with the selected hull vertices, ``weights=None``
            (to be filled by an external representation model), and the
            final projection error in ``scores``.
        """
        Z = context.df_features.values
        labels = list(context.df_features.index)
        N, p = Z.shape

        selected_idx: list[int] = []
        remaining_idx = list(range(N))

        for _ in range(self.k):
            best_candidate = None
            best_error = np.inf

            for c in remaining_idx:
                candidate_idx = selected_idx + [c]
                error = self._total_projection_error(Z, candidate_idx)
                if error < best_error:
                    best_error = error
                    best_candidate = c

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)

        final_error = self._total_projection_error(Z, selected_idx)
        selection = tuple(labels[i] for i in selected_idx)

        slice_labels = context.slicer.labels_for_index(context.df_raw.index)
        representatives = {
            s: context.df_raw.loc[slice_labels == s] for s in selection
        }

        return RepSetResult(
            context=context,
            selection_space='subset',
            selection=selection,
            scores={'projection_error': final_error},
            representatives=representatives,
            weights=None,
        )

    def _total_projection_error(
        self, Z: np.ndarray, selected_idx: list[int]
    ) -> float:
        """Compute total reconstruction error across all periods.

        Args:
            Z: Full feature matrix (N x p).
            selected_idx: Indices of currently selected hull vertices.

        Returns:
            Sum of squared reconstruction errors over all N periods.
        """
        Z_sel = Z[selected_idx]
        total_error = 0.0
        for i in range(Z.shape[0]):
            total_error += self._projection_error(Z[i], Z_sel)
        return total_error

    def _projection_error(self, z: np.ndarray, Z_sel: np.ndarray) -> float:
        """Compute minimum reconstruction error for a single period.

        Args:
            z: Feature vector for one period (length p).
            Z_sel: Feature matrix of selected hull vertices (k_current x p).

        Returns:
            Squared L2 reconstruction error.
        """
        if self.hull_type == 'conic':
            return self._projection_error_conic(z, Z_sel)
        return self._projection_error_convex(z, Z_sel)

    def _projection_error_conic(
        self, z: np.ndarray, Z_sel: np.ndarray
    ) -> float:
        """Conic projection: min_w ||z - Z_sel^T @ w||^2, w >= 0."""
        w, residual = nnls(Z_sel.T, z)
        reconstruction = Z_sel.T @ w
        return float(np.sum((z - reconstruction) ** 2))

    def _projection_error_convex(
        self, z: np.ndarray, Z_sel: np.ndarray
    ) -> float:
        """Convex projection: min_w ||z - Z_sel^T @ w||^2, w >= 0, sum(w) = 1."""
        k_sel = Z_sel.shape[0]
        if k_sel == 1:
            return float(np.sum((z - Z_sel[0]) ** 2))

        def objective(w):
            return np.sum((z - w @ Z_sel) ** 2)

        w0 = np.ones(k_sel) / k_sel
        bounds = [(0.0, 1.0)] * k_sel
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        result = minimize(
            objective, w0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 200},
        )
        return float(result.fun)
