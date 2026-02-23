from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

import numpy as np
from sklearn_extra.cluster import KMedoids

from .search_algorithm import SearchAlgorithm
from ..results import RepSetResult

if TYPE_CHECKING:
    from ..context import ProblemContext


class KMedoidsSearch(SearchAlgorithm):
    """K-medoids clustering for representative subset selection.

    Wraps ``sklearn_extra.cluster.KMedoids`` to partition feature-space
    slices into k clusters and select the medoid of each cluster as a
    representative period. Weights are computed as the fraction of slices
    assigned to each cluster.

    This is a constructive (Workflow Type 2) algorithm: it has its own
    internal objective and does not require an external ``ObjectiveSet``.
    The ``RepresentationModel`` is skipped by ``RepSetExperiment.run()``
    because weights are pre-computed.

    Args:
        k: Number of clusters / representative periods.
        metric: Distance metric for k-medoids (default ``'euclidean'``).
        method: K-medoids algorithm variant. ``'alternate'`` (default) or
            ``'pam'`` (Partitioning Around Medoids, slower but optimal).
        init: Initialization method (default ``'k-medoids++'``).
        random_state: Seed for reproducibility.
        max_iter: Maximum number of iterations.

    Examples:
        Basic usage:

        >>> from energy_repset.search_algorithms import KMedoidsSearch
        >>> search = KMedoidsSearch(k=4, random_state=42)
        >>> result = search.find_selection(feature_context)
        >>> result.selection  # Tuple of medoid labels
        >>> result.weights    # Dict mapping labels to cluster-size fractions
    """

    def __init__(
        self,
        k: int,
        metric: str = 'euclidean',
        method: str = 'alternate',
        init: str = 'k-medoids++',
        random_state: int | None = None,
        max_iter: int = 300,
    ):
        """Initialize k-medoids clustering search.

        Args:
            k: Number of clusters to produce.
            metric: Distance metric passed to ``KMedoids``.
            method: Algorithm variant (``'alternate'`` or ``'pam'``).
            init: Medoid initialization strategy.
            random_state: Random seed for reproducibility.
            max_iter: Maximum iterations for convergence.
        """
        self.k = k
        self.metric = metric
        self.method = method
        self.init = init
        self.random_state = random_state
        self.max_iter = max_iter

    def find_selection(self, context: ProblemContext) -> RepSetResult:
        """Run k-medoids clustering on the feature space.

        Args:
            context: Problem context with ``df_features`` populated.

        Returns:
            RepSetResult with medoid labels as the selection, pre-computed
            cluster-size-proportional weights, and WCSS (Within-Cluster Sum
            of Squares) in ``scores``.
        """
        Z = context.df_features.values
        labels = list(context.df_features.index)

        model = KMedoids(
            n_clusters=self.k,
            metric=self.metric,
            method=self.method,
            init=self.init,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        model.fit(Z)

        selection, weights, wcss, diagnostics = self._extract_results(
            Z, labels, model.labels_, model.medoid_indices_
        )
        diagnostics['inertia'] = float(model.inertia_)
        diagnostics['n_iter'] = int(model.n_iter_)

        slice_labels = context.slicer.labels_for_index(context.df_raw.index)
        representatives = {
            s: context.df_raw.loc[slice_labels == s] for s in selection
        }

        return RepSetResult(
            context=context,
            selection_space='subset',
            selection=selection,
            scores={'wcss': wcss},
            representatives=representatives,
            weights=weights,
            diagnostics=diagnostics,
        )

    def _extract_results(
        self,
        Z: np.ndarray,
        labels: list,
        cluster_labels: np.ndarray,
        medoid_indices: np.ndarray,
    ) -> tuple:
        """Extract selection, weights, WCSS, and diagnostics from clustering.

        Args:
            Z: Feature matrix (N x p).
            labels: Slice labels aligned with rows of Z.
            cluster_labels: Cluster assignment for each slice (length N).
            medoid_indices: Row indices of medoids in Z.

        Returns:
            Tuple of (selection, weights, wcss, diagnostics).
        """
        unique_clusters = sorted(set(cluster_labels))
        N = len(labels)
        wcss = 0.0

        selected_labels = []
        weight_dict: Dict[Any, float] = {}
        cluster_info: list[Dict[str, Any]] = []

        for c in unique_clusters:
            mask = cluster_labels == c
            indices = np.where(mask)[0]
            cluster_Z = Z[indices]
            centroid = cluster_Z.mean(axis=0)

            cluster_wcss = float(np.sum((cluster_Z - centroid) ** 2))
            wcss += cluster_wcss

            medoid_idx = medoid_indices[c]
            rep_label = labels[medoid_idx]

            fraction = len(indices) / N
            selected_labels.append(rep_label)
            weight_dict[rep_label] = fraction

            member_labels = [labels[i] for i in indices]
            cluster_info.append({
                'cluster': int(c),
                'medoid': rep_label,
                'size': len(indices),
                'members': member_labels,
            })

        selection = tuple(selected_labels)
        diagnostics = {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_info': cluster_info,
        }

        return selection, weight_dict, float(wcss), diagnostics
