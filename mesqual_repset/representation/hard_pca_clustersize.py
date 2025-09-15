from __future__ import annotations

from typing import Dict, Hashable, TYPE_CHECKING

from .representation import RepresentationModel

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class HardPCAClustersizeRepresentation(RepresentationModel):
    """Assigns weights based on cluster size (hard assignment)."""

    def fit(self, context: ProblemContext):
        """Stores the full feature matrix for later use."""
        self.all_features_ = context.df_features
        self.all_slice_labels_ = context.candidates

    def weigh(self, combination: SliceCombination) -> Dict[Hashable, float]:
        if not combination:
            return {}

        # --- Use data stored during fit ---
        # 1. Get feature vectors for the selection from self.all_features_.
        # 2. Assign each slice in self.all_slice_labels_ to its
        #    closest representative in the selection.
        # 3. Count cluster sizes and normalize to get weights.
        # 4. Return dictionary of {representative_label: weight}.

        # Dummy implementation for illustration
        total_slices = len(self.all_slice_labels_)
        weight = 1.0 / len(combination) if combination else 0
        dummy_weights = {label: weight for label in combination}
        return dummy_weights
