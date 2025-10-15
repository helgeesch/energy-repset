from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

if TYPE_CHECKING:
    from .feature_engineering import FeatureEngineer
    from .search_algorithms.search_algorithm import SearchAlgorithm
    from .representation import RepresentationModel


@dataclass
class Workflow:
    """A serializable object that defines a complete selection problem.

    This dataclass encapsulates all components needed to execute a representative
    subset selection workflow: feature engineering, search algorithm, representation
    model, and the target number of periods to select.

    Attributes:
        feature_engineer: Component that transforms raw time-series into features.
        search_algorithm: Algorithm that finds the optimal subset of k periods.
        representation_model: Model that calculates responsibility weights for
            selected periods.
        k: Number of representative periods to select.

    Examples:
        Define a complete workflow:

        >>> from mesqual_repset.workflow import Workflow
        >>> from mesqual_repset.feature_engineering import StandardStatsFeatureEngineer
        >>> from mesqual_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
        >>> from mesqual_repset.representation import UniformRepresentationModel
        >>> from mesqual_repset.objectives import ObjectiveSet
        >>> from mesqual_repset.score_components import WassersteinFidelity
        >>> from mesqual_repset.selection_policies import ParetoMaxMinStrategy
        >>> from mesqual_repset.combination_generator import ExhaustiveCombinationGenerator
        >>>
        >>> # Create components
        >>> feature_eng = StandardStatsFeatureEngineer()
        >>> objective_set = ObjectiveSet({'wass': (1.0, WassersteinFidelity())})
        >>> policy = ParetoMaxMinStrategy()
        >>> combo_gen = ExhaustiveCombinationGenerator(k=3)
        >>> search_algo = ObjectiveDrivenCombinatorialSearchAlgorithm(
        ...     objective_set, policy, combo_gen
        ... )
        >>> repr_model = UniformRepresentationModel()
        >>>
        >>> # Create workflow
        >>> workflow = Workflow(
        ...     feature_engineer=feature_eng,
        ...     search_algorithm=search_algo,
        ...     representation_model=repr_model,
        ...     k=3
        ... )
    """
    feature_engineer: FeatureEngineer
    search_algorithm: SearchAlgorithm
    representation_model: RepresentationModel
    k: int

    def save(self, filepath: str | Path):
        """Save workflow configuration to file.

        Args:
            filepath: Path where workflow configuration will be saved.

        Note:
            Not yet implemented. Future implementation will serialize to JSON/YAML.
        """
        # Logic to serialize the workflow config to JSON/YAML
        pass

    @classmethod
    def load(cls, filepath: str | Path):
        """Load workflow configuration from file.

        Args:
            filepath: Path to workflow configuration file.

        Returns:
            Reconstructed Workflow instance.

        Note:
            Not yet implemented. Future implementation will deserialize from JSON/YAML.
        """
        # Logic to deserialize and reconstruct the workflow
        pass
