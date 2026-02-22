from __future__ import annotations

from typing import Literal, Dict, Any, TYPE_CHECKING

import numpy as np
from scipy.sparse import diags
from sklearn.cluster import AgglomerativeClustering

from .search_algorithm import SearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..context import ProblemContext


class CTPCSearch(SearchAlgorithm):
    """Chronological Time-Period Clustering with contiguity constraint.

    Implements hierarchical agglomerative clustering where only temporally
    adjacent periods may merge, producing k contiguous time segments.
    Based on Pineda & Morales (2018).

    The algorithm computes weights as the fraction of time covered by each
    segment, so the external representation model is skipped when the result
    is used in ``RepSetExperiment.run()``.

    Args:
        k: Number of contiguous time segments to produce.
        linkage: Linkage criterion for agglomerative clustering. One of
            ``'ward'``, ``'complete'``, ``'average'``, or ``'single'``.

    References:
        S. Pineda, J. M. Morales. "Chronological Time-Period Clustering
        for Optimal Capacity Expansion Planning With Storage."
        IEEE Trans. Power Syst., 33(6), 7162--7170, 2018.

    Examples:
        Basic usage:

        >>> from energy_repset.search_algorithms import CTPCSearch
        >>> search = CTPCSearch(k=4, linkage='ward')
        >>> result = search.find_selection(feature_context)
        >>> result.selection  # Tuple of medoid labels
        >>> result.weights    # Dict mapping labels to time fractions
    """

    def __init__(
        self,
        k: int,
        linkage: Literal['ward', 'complete', 'average', 'single'] = 'ward',
    ):
        """Initialize CTPC search.

        Args:
            k: Number of contiguous clusters to produce.
            linkage: Agglomerative linkage criterion.
        """
        self._k = k
        self.linkage = linkage

    @property
    def k(self) -> int:
        """Number of contiguous time segments to produce."""
        return self._k

    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Run contiguity-constrained hierarchical clustering.

        Args:
            context: Problem context with ``df_features`` populated. Slices
                must be naturally ordered by time (which they are when coming
                from ``TimeSlicer``).

        Returns:
            RepSetResult with medoid (or centroid) labels as the selection,
            pre-computed weights (segment size fractions), and within-cluster
            sum of squares in ``scores``.
        """
        Z = context.df_features.values
        labels = list(context.df_features.index)
        N = Z.shape[0]

        connectivity = self._build_connectivity(N)

        clustering = AgglomerativeClustering(
            n_clusters=self.k,
            connectivity=connectivity,
            linkage=self.linkage,
        )
        cluster_labels = clustering.fit_predict(Z)

        selection, weights, wcss, diagnostics = self._extract_results(
            Z, labels, cluster_labels
        )

        slice_labels = context.slicer.labels_for_index(context.df_raw.index)
        representatives = {
            s: context.df_raw.loc[slice_labels == s] for s in selection
        }

        return RepSetResult(
            context=context,
            selection_space='chronological',
            selection=selection,
            scores={'wcss': wcss},
            representatives=representatives,
            weights=weights,
            diagnostics=diagnostics,
        )

    def _build_connectivity(self, n: int) -> np.ndarray:
        """Build tridiagonal connectivity matrix for n slices.

        Args:
            n: Number of time slices.

        Returns:
            Sparse-like (n x n) binary adjacency matrix connecting only
            temporally adjacent slices.
        """
        off_diag = np.ones(n - 1)
        return diags([off_diag, np.ones(n), off_diag], [-1, 0, 1]).toarray()

    def _extract_results(
        self,
        Z: np.ndarray,
        labels: list,
        cluster_labels: np.ndarray,
    ) -> tuple:
        """Extract selection, weights, WCSS, and diagnostics from clustering.

        Args:
            Z: Feature matrix (N x p).
            labels: Slice labels aligned with rows of Z.
            cluster_labels: Cluster assignment for each slice (length N).

        Returns:
            Tuple of (selection, weights, wcss, diagnostics).
        """
        unique_clusters = sorted(set(cluster_labels))
        N = len(labels)
        wcss = 0.0

        selected_labels = []
        weight_dict: Dict = {}
        segment_info: list[Dict[str, Any]] = []

        for c in unique_clusters:
            mask = cluster_labels == c
            indices = np.where(mask)[0]
            cluster_Z = Z[indices]
            centroid = cluster_Z.mean(axis=0)

            cluster_wcss = np.sum((cluster_Z - centroid) ** 2)
            wcss += cluster_wcss

            dists = np.sum((cluster_Z - centroid) ** 2, axis=1)
            medoid_local = int(np.argmin(dists))
            rep_idx = indices[medoid_local]
            rep_label = labels[rep_idx]

            fraction = len(indices) / N
            selected_labels.append(rep_label)
            weight_dict[rep_label] = fraction

            segment_info.append({
                'cluster': c,
                'start': labels[indices[0]],
                'end': labels[indices[-1]],
                'size': len(indices),
                'representative': rep_label,
            })

        segment_info.sort(key=lambda seg: seg['start'])

        selection = tuple(selected_labels)
        diagnostics = {
            'cluster_labels': cluster_labels.tolist(),
            'segments': segment_info,
        }

        return selection, weight_dict, float(wcss), diagnostics
