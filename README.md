# energy-repset

A unified, modular framework for **representative subset selection in multi-variate time-series spaces**.

Select a small number of representative periods (e.g., weeks or months) from a full year of data to approximate the full dataset in downstream modeling tasks like energy systems optimization.

## The Five-Component Framework

The framework decomposes any time-series aggregation method into five interchangeable components:

| Component | Symbol | Role |
|-----------|--------|------|
| Feature Space | **F** | How raw time-series are transformed into comparable representations |
| Objective | **O** | How candidate selections are scored for quality |
| Selection Space | **S** | What is being selected (historical subsets, synthetic archetypes, etc.) |
| Representation Model | **R** | How selected periods represent the full dataset |
| Search Algorithm | **A** | The engine that finds optimal selections |

Any concrete method -- k-means, k-medoids, MILP-based selection, genetic algorithms -- is a specific instantiation of this five-component structure. For the full theoretical treatment, see the [Unified Framework](https://helgeesch.github.io/energy-repset/unified_framework/) document.

## Installation

```bash
pip install energy-repset
```

## Quick Start

```python
import pandas as pd
from energy_repset.context import ProblemContext
from energy_repset.time_slicer import TimeSlicer
from energy_repset.feature_engineering import StandardStatsFeatureEngineer
from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity, CorrelationFidelity
from energy_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from energy_repset.selection_policies import WeightedSumPolicy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.representation import UniformRepresentationModel
from energy_repset.problem import RepSetExperiment
from energy_repset.workflow import Workflow

# Load hourly time-series data (columns = variables, index = datetime)
df_raw = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Define problem: slice the year into monthly candidate periods
slicer = TimeSlicer(unit="month")
context = ProblemContext(df_raw, slicer)

# Feature engineering: statistical summaries per month
feature_engineer = StandardStatsFeatureEngineer()

# Objective: score each candidate selection on distribution fidelity
objective_set = ObjectiveSet({
    'wasserstein': (1.0, WassersteinFidelity()),
    'correlation': (1.0, CorrelationFidelity()),
})

# Search: evaluate all 4-of-12 monthly combinations
policy = WeightedSumPolicy()
combi_gen = ExhaustiveCombiGen(k=4)
search = ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combi_gen)

# Representation: equal 1/k weights per selected month
representation = UniformRepresentationModel()

# Assemble and run
workflow = Workflow(feature_engineer, search, representation)
experiment = RepSetExperiment(context, workflow)
result = experiment.run()

print(result.selection)  # e.g., (Period('2019-01', 'M'), Period('2019-04', 'M'), ...)
print(result.weights)    # e.g., {Period('2019-01', 'M'): 3.0, ...}
print(result.scores)     # e.g., {'wasserstein': 0.023, 'correlation': 0.015}
```

## Documentation

- [Getting Started](https://helgeesch.github.io/energy-repset/getting_started/) -- end-to-end walkthrough
- [Unified Framework](https://helgeesch.github.io/energy-repset/unified_framework/) -- theoretical foundations
- [Workflow Types](https://helgeesch.github.io/energy-repset/workflow/) -- generate-and-test, constructive, and direct optimization
- [Modules & Components](https://helgeesch.github.io/energy-repset/modules/) -- all implementations and their APIs
- [Gallery](https://helgeesch.github.io/energy-repset/gallery/) -- interactive examples

## License

Apache-2.0
