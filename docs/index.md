# energy-repset

A unified, modular framework for **representative subset selection in multi-variate time-series spaces**.

## Overview

Select a small number of representative periods (e.g., weeks or months) from a full year of data to approximate the full dataset in downstream modeling tasks like energy systems optimization.

The framework decomposes any selection method into five interchangeable pillars:

| Pillar | Symbol | Role |
|--------|--------|------|
| Feature Space | **F** | How raw time-series are transformed into comparable representations |
| Objective | **O** | How candidate selections are scored for quality |
| Selection Space | **S** | What is being selected (historical subsets, synthetic archetypes, etc.) |
| Representation Model | **R** | How selected periods represent the full dataset |
| Search Algorithm | **A** | The engine that finds optimal selections |

## Quick Start

```bash
pip install energy-repset
```

```python
from energy_repset.context import ProblemContext
from energy_repset.time_slicer import TimeSlicer
from energy_repset.feature_engineering import StandardStatsFeatureEngineer
from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, CorrelationFidelity
from energy_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from energy_repset.selection_policies import ParetoMaxMinStrategy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.representation import KMedoidsClustersizeRepresentation
from energy_repset.problem import RepSetExperiment
from energy_repset.workflow import Workflow

# Define problem
slicer = TimeSlicer(unit="month")
context = ProblemContext(df_raw, slicer)

# Configure pipeline
feature_engineer = StandardStatsFeatureEngineer()
objective_set = ObjectiveSet.from_components(
    WassersteinFidelity(direction="min"),
    CorrelationFidelity(direction="min"),
)
search = ObjectiveDrivenCombinatorialSearchAlgorithm(
    combination_generator=ExhaustiveCombiGen(),
    objective_set=objective_set,
    selection_policy=ParetoMaxMinStrategy(),
)
representation = KMedoidsClustersizeRepresentation()
workflow = Workflow(feature_engineer, search, representation, k=4)

# Run
experiment = RepSetExperiment(context, workflow)
result = experiment.run()
```
