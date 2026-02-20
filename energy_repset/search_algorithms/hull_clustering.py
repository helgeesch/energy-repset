from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize, nnls

from .search_algorithm import SearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..context import ProblemContext


class HullClusteringSearch(SearchAlgorithm):
    """Farthest-point greedy hull clustering (Neustroev et al., 2025).

    Implements the greedy convex/conic hull clustering algorithm from
    Neustroev et al. (2025).  At each iteration the algorithm selects
    the data point **furthest from the current hull**, i.e. the point
    with maximum projection error onto the hull spanned by the already-
    selected representatives.  The first representative is the point
    furthest from the dataset mean.

    This farthest-point strategy naturally selects extreme/boundary
    periods first, producing a hull that spans the data well.

    The algorithm leaves ``weights=None`` in the result so that an external
    ``RepresentationModel`` (typically ``BlendedRepresentationModel``) can
    compute the final soft-assignment weights.

    Args:
        k: Number of representative periods to select.
        hull_type: Type of projection constraint. ``'convex'`` enforces
            non-negative weights that sum to 1. ``'conic'`` enforces only
            non-negativity.

    References:
        G. Neustroev, D. A. Tejada-Arango, G. Morales-Espana,
        M. M. de Weerdt. "Hull Clustering with Blended Representative
        Periods for Energy System Optimization Models."
        arXiv:2508.21641, 2025.

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
        """Find k hull vertices via farthest-point greedy selection.

        The algorithm (Algorithm 2 in Neustroev et al.):

        1. Select the point furthest from the dataset mean.
        2. For iterations 2..k, compute the projection error (hull
           distance) for every remaining point and select the one with
           the **maximum** error.

        Args:
            context: Problem context with ``df_features`` populated.

        Returns:
            RepSetResult with the selected hull vertices, ``weights=None``
            (to be filled by an external representation model), and the
            final projection error in ``scores``.
        """
        Z = context.df_features.values
        labels = list(context.df_features.index)
        N = Z.shape[0]

        selected_idx, final_error = self._greedy_farthest_point(Z)

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

    def _greedy_farthest_point(
        self, Z: np.ndarray
    ) -> tuple[list[int], float]:
        """Run the farthest-point greedy hull clustering.

        Args:
            Z: Feature matrix (N x p).

        Returns:
            Tuple of (selected indices, total projection error).
        """
        N = Z.shape[0]

        first_idx = self._init_furthest_from_mean(Z)
        selected_idx = [first_idx]
        remaining = set(range(N)) - {first_idx}

        hull_dists = np.full(N, np.inf)
        self._update_hull_distances(Z, selected_idx, remaining, hull_dists)

        for _ in range(self.k - 1):
            best = max(remaining, key=lambda i: hull_dists[i])
            selected_idx.append(best)
            remaining.discard(best)

            if remaining:
                self._update_hull_distances(
                    Z, selected_idx, remaining, hull_dists
                )

        final_error = sum(hull_dists[i] for i in range(N) if i not in selected_idx)
        return selected_idx, float(final_error)

    @staticmethod
    def _init_furthest_from_mean(Z: np.ndarray) -> int:
        """Select the point furthest from the dataset mean.

        Args:
            Z: Feature matrix (N x p).

        Returns:
            Index of the point with maximum squared distance to the mean.
        """
        mean_z = Z.mean(axis=0)
        dists = np.sum((Z - mean_z) ** 2, axis=1)
        return int(np.argmax(dists))

    def _update_hull_distances(
        self,
        Z: np.ndarray,
        selected_idx: list[int],
        remaining: set[int],
        hull_dists: np.ndarray,
    ) -> None:
        """Recompute hull distances for remaining points.

        Args:
            Z: Feature matrix (N x p).
            selected_idx: Currently selected hull vertex indices.
            remaining: Set of indices still available for selection.
            hull_dists: Array to update in-place with new hull distances.
        """
        Z_sel = Z[selected_idx]
        for i in remaining:
            hull_dists[i] = self._projection_error(Z[i], Z_sel)

    def _projection_error(self, z: np.ndarray, Z_sel: np.ndarray) -> float:
        """Compute projection error for a single point onto the hull.

        Args:
            z: Feature vector for one period (length p).
            Z_sel: Feature matrix of selected hull vertices (k_current x p).

        Returns:
            Squared L2 projection error.
        """
        if self.hull_type == 'conic':
            return self._projection_error_conic(z, Z_sel)
        return self._projection_error_convex(z, Z_sel)

    def _projection_error_conic(
        self, z: np.ndarray, Z_sel: np.ndarray
    ) -> float:
        """Conic projection: min_w ||z - Z_sel^T @ w||^2, w >= 0."""
        w, _ = nnls(Z_sel.T, z)
        reconstruction = Z_sel.T @ w
        return float(np.sum((z - reconstruction) ** 2))

    def _projection_error_convex(
        self, z: np.ndarray, Z_sel: np.ndarray
    ) -> float:
        """Convex projection: min_w ||z - w @ Z_sel||^2, w >= 0, sum(w) = 1."""
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
