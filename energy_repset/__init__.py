"""energy-repset: representative subset selection for multi-variate time-series.

Usage::

    import energy_repset as rep
    import energy_repset.diagnostics as diag
"""

from .context import ProblemContext
from .time_slicer import TimeSlicer
from .workflow import Workflow
from .problem import RepSetExperiment
from .results import RepSetResult
from .objectives import ObjectiveSet, ObjectiveSpec

from .feature_engineering import (
    FeatureEngineer,
    FeaturePipeline,
    StandardStatsFeatureEngineer,
    PCAFeatureEngineer,
)

from .score_components import (
    WassersteinFidelity,
    CorrelationFidelity,
    DiurnalFidelity,
    DiversityReward,
    CentroidBalance,
    CoverageBalance,
    NRMSEFidelity,
    DTWFidelity,
    DurationCurveFidelity,
    DiurnalDTWFidelity,
)

from .combi_gens import (
    CombinationGenerator,
    ExhaustiveCombiGen,
    GroupQuotaCombiGen,
    ExhaustiveHierarchicalCombiGen,
    GroupQuotaHierarchicalCombiGen,
)

from .selection_policies import (
    SelectionPolicy,
    PolicyOutcome,
    WeightedSumPolicy,
    ParetoUtopiaPolicy,
    ParetoMaxMinStrategy,
    ParetoOutcome,
)

from .search_algorithms import (
    SearchAlgorithm,
    ObjectiveDrivenSearchAlgorithm,
    ObjectiveDrivenCombinatorialSearchAlgorithm,
    CTPCSearch,
)

from .representation import (
    RepresentationModel,
    UniformRepresentationModel,
    KMedoidsClustersizeRepresentation,
    BlendedRepresentationModel,
)

__all__ = [
    # Core
    "ProblemContext",
    "TimeSlicer",
    "Workflow",
    "RepSetExperiment",
    "RepSetResult",
    "ObjectiveSet",
    "ObjectiveSpec",
    # Feature engineering
    "FeatureEngineer",
    "FeaturePipeline",
    "StandardStatsFeatureEngineer",
    "PCAFeatureEngineer",
    # Score components
    "WassersteinFidelity",
    "CorrelationFidelity",
    "DiurnalFidelity",
    "DiversityReward",
    "CentroidBalance",
    "CoverageBalance",
    "NRMSEFidelity",
    "DTWFidelity",
    "DurationCurveFidelity",
    "DiurnalDTWFidelity",
    # Combination generators
    "CombinationGenerator",
    "ExhaustiveCombiGen",
    "GroupQuotaCombiGen",
    "ExhaustiveHierarchicalCombiGen",
    "GroupQuotaHierarchicalCombiGen",
    # Selection policies
    "SelectionPolicy",
    "PolicyOutcome",
    "WeightedSumPolicy",
    "ParetoUtopiaPolicy",
    "ParetoMaxMinStrategy",
    "ParetoOutcome",
    # Search algorithms
    "SearchAlgorithm",
    "ObjectiveDrivenSearchAlgorithm",
    "ObjectiveDrivenCombinatorialSearchAlgorithm",
    "CTPCSearch",
    # Representation models
    "RepresentationModel",
    "UniformRepresentationModel",
    "KMedoidsClustersizeRepresentation",
    "BlendedRepresentationModel",
]
