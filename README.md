# energy-repset

A unified, modular framework for **representative subset selection in multi-variate time-series spaces**.

## Why this package?

Energy system models, capacity expansion studies, and other time-series-heavy applications often need to reduce a full year (or longer) of hourly data to a small set of representative periods -- days, weeks, or months -- without losing what matters. The literature offers many methods (k-means, k-medoids, MILP-based selection, genetic algorithms, etc.), but the landscape is dense and tangled: each method bundles multiple decisions -- how to represent data, what to optimize, how to search -- into a single procedure, making it hard to see which choices matter, compare approaches on equal footing, or adapt a method to your specific problem.

`energy-repset` clears a path through the jungle in two ways:

1. **A unified framework** that decomposes *any* time-series aggregation method into five interchangeable components. Every established methodology is a specific instantiation of this structure. The framework provides a common language for describing, comparing, and assembling methods. The full theoretical treatment is available in the [Unified Framework](https://helgeesch.github.io/energy-repset/unified_framework/) document.

2. **A modular Python package** that implements this framework as a library of composable, protocol-based modules. You pick one implementation per component, wire them together, and run. Adding a new algorithm or score metric means implementing a single protocol -- everything else stays the same.

## The Five Components

| Component | Symbol | Role |
|-----------|--------|------|
| Feature Space | **F** | How raw time-series are transformed into comparable representations |
| Objective | **O** | How candidate selections are scored for quality |
| Selection Space | **S** | What is being selected (historical subsets, synthetic archetypes, etc.) |
| Representation Model | **R** | How selected periods represent the full dataset |
| Search Algorithm | **A** | The engine that finds optimal selections |

## Navigating the project

**Documentation site** ([helgeesch.github.io/energy-repset](https://helgeesch.github.io/energy-repset/)):

| Section | What you'll find |
|---------|-----------------|
| [Unified Framework](https://helgeesch.github.io/energy-repset/unified_framework/) | The theoretical paper: problem decomposition, component taxonomy, method comparison |
| [Workflow Types](https://helgeesch.github.io/energy-repset/workflow/) | The three workflow patterns: generate-and-test, constructive, direct optimization |
| [Modules & Components](https://helgeesch.github.io/energy-repset/modules/) | Inventory of all implemented modules and how they map to the five components |
| [Configuration Advisor](https://helgeesch.github.io/energy-repset/advisor/) | Decision guide for choosing components based on your problem |
| [Getting Started](https://helgeesch.github.io/energy-repset/getting_started/) | End-to-end walkthrough from data to result |
| [Examples](https://helgeesch.github.io/energy-repset/gallery/) | Worked examples showcasing different configurations |
| [API Reference](https://helgeesch.github.io/energy-repset/api/) | Auto-generated class and method documentation |

**Package structure** (`energy_repset/`):

| Module | Framework component |
|--------|-------------------|
| `context`, `time_slicer` | Problem definition and data container |
| `feature_engineering/` | **F** -- Feature engineers (statistical summaries, PCA, pipelines) |
| `objectives`, `score_components/` | **O** -- Objective sets and scoring metrics |
| `combi_gens/` | **S** -- Combination generators (exhaustive, group-quota, hierarchical) |
| `representation/` | **R** -- Representation models (uniform, cluster-based, blended) |
| `search_algorithms/`, `selection_policies/` | **A** -- Search algorithms and selection policies |
| `workflow`, `problem`, `results` | Orchestration: wire components, run, collect results |
| `diagnostics/` | Visualization and analysis of features, scores, and results |

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

## License

Apache-2.0
