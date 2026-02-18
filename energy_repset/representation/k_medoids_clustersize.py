from __future__ import annotations

from typing import Dict, Hashable, TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist

from .representation import RepresentationModel

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class KMedoidsClustersizeRepresentation(RepresentationModel):
    """Assigns weights based on k-medoids cluster sizes (hard assignment).

    This representation model performs virtual k-medoids clustering where the
    selected periods are enforced as medoids (cluster centers). Each candidate
    period is assigned to its nearest medoid, and weights are calculated as
    the proportion of periods assigned to each medoid.

    The weights reflect how many original periods each representative is
    responsible for, making this appropriate when representatives should be
    weighted by their "sphere of influence" in feature space.

    Attributes:
        all_features_: Feature matrix for all candidate periods (set during fit).
        all_slice_labels_: Labels for all candidate periods (set during fit).

    Examples:

        >>> model = KMedoidsClustersizeRepresentation()
        >>> model.fit(context)  # context has 12 monthly candidates
        >>> weights = model.weigh((Period('2024-01', 'M'), Period('2024-06', 'M')))
        >>> print(weights)
            {Period('2024-01', 'M'): 0.583, Period('2024-06', 'M'): 0.417}
        >>> # Jan represents 7 months, Jun represents 5 months
    """

    def fit(self, context: ProblemContext):
        """Store the full feature matrix for later clustering.

        Args:
            context: Problem context containing df_features and candidates.
        """
        self.all_features_ = context.df_features
        self.all_slice_labels_ = context.slicer.unique_slices(context.df_raw.index)

    def weigh(self, combination: SliceCombination) -> Dict[Hashable, float]:
        """Calculate weights based on cluster sizes from hard assignment.

        Performs virtual k-medoids clustering where:
        1. Selected periods are enforced as medoids
        2. Each candidate is assigned to its nearest medoid (Euclidean distance)
        3. Weight = (cluster size) / (total candidates)

        Args:
            combination: Tuple of selected slice identifiers.

        Returns:
            Dictionary mapping each slice ID to its weight (proportion of
            candidates assigned to it).

        Raises:
            ValueError: If combination contains slices not in the feature matrix.
        """
        if not combination:
            return {}

        # Extract feature vectors for selected medoids
        medoid_indices = []
        for slice_label in combination:
            if slice_label not in self.all_slice_labels_:
                raise ValueError(f"Slice {slice_label} not found in candidates")
            medoid_indices.append(self.all_slice_labels_.index(slice_label))

        medoid_features = self.all_features_.iloc[medoid_indices].values
        all_features = self.all_features_.values

        # Compute pairwise distances: shape (n_candidates, k_medoids)
        distances = cdist(all_features, medoid_features, metric='euclidean')

        # Hard assignment: each candidate assigned to nearest medoid
        assignments = np.argmin(distances, axis=1)

        # Count cluster sizes
        cluster_sizes = np.bincount(assignments, minlength=len(combination))

        # Calculate weights as proportions
        total_candidates = len(self.all_slice_labels_)
        weights = {
            slice_label: float(cluster_sizes[i]) / total_candidates
            for i, slice_label in enumerate(combination)
        }

        return weights
