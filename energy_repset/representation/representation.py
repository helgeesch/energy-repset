from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Dict, Hashable

import pandas as pd

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class RepresentationModel(ABC):
    """Base class for representation models (Pillar R).

    Defines how selected representative periods represent the full dataset by
    calculating responsibility weights. The model is first fitted to learn about
    the entire dataset, then the weigh() method calculates weights for specific
    selections.

    Different models implement different weighting strategies:
    - Uniform: Equal weights (e.g., 365/k for yearly data)
    - Cluster-based: Weights proportional to cluster sizes
    - Blended: Soft assignment where each period is a weighted mix of representatives

    Examples:
        >>> class UniformWeights(RepresentationModel):
        ...     def fit(self, context: ProblemContext):
        ...         self.n_total = len(context.slicer.slices)
        ...
        ...     def weigh(self, combination: SliceCombination) -> Dict[Hashable, float]:
        ...         weight = self.n_total / len(combination)
        ...         return {slice_id: weight for slice_id in combination}
        ...
        >>> model = UniformWeights()
        >>> model.fit(context)
        >>> weights = model.weigh((0, 3, 6, 9))
        >>> print(weights)  # {0: 91.25, 3: 91.25, 6: 91.25, 9: 91.25} for 365 days, k=4

        >>> class ClusterSizeWeights(RepresentationModel):
        ...     def fit(self, context: ProblemContext):
        ...         from sklearn.cluster import KMeans
        ...         self.kmeans = KMeans(n_clusters=4)
        ...         self.kmeans.fit(context.df_features)
        ...
        ...     def weigh(self, combination: SliceCombination) -> Dict[Hashable, float]:
        ...         labels = self.kmeans.labels_
        ...         weights = {}
        ...         for i, slice_id in enumerate(combination):
        ...             cluster_size = (labels == i).sum()
        ...             weights[slice_id] = cluster_size
        ...         return weights
        ...
        >>> model = ClusterSizeWeights()
        >>> model.fit(context)
        >>> weights = model.weigh((0, 3, 6, 9))
        >>> print(weights)  # Weights proportional to cluster membership
    """

    @abstractmethod
    def fit(self, context: 'ProblemContext'):
        """Fit the representation model to the full dataset.

        This method performs any necessary pre-computation based on the full set
        of candidate slices (e.g., storing the feature matrix, fitting clustering
        models, computing distance matrices).

        Args:
            context: The problem context with df_features populated. Feature
                engineering must be run before calling this method.
        """
        ...

    @abstractmethod
    def weigh(
        self,
        combination: SliceCombination
    ) -> Union[Dict[Hashable, float], pd.DataFrame]:
        """Calculate representation weights for a given selection.

        This method should only be called after the model has been fitted.

        Args:
            combination: Tuple of selected slice identifiers for which to
                calculate representation weights.

        Returns:
            The calculated weights, either as a dictionary mapping each selected
            slice to its weight, or as a DataFrame for more complex weight
            structures (e.g., blended models where each original period has
            weights across multiple representatives).
        """
        ...
