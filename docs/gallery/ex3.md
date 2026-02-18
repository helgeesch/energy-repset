# Example 3: Getting Started

The simplest possible end-to-end workflow. Selects 4 representative months
from a year of hourly time-series data using a single objective (Wasserstein
fidelity), a weighted-sum policy, and uniform weights. A minimal "hello world"
for onboarding.

**Script:** [`examples/ex3_getting_started.py`](https://github.com/helgeesch/energy-repset/blob/main/examples/ex3_getting_started.py)

| Pillar | Component |
|--------|-----------|
| F | `StandardStatsFeatureEngineer` |
| O | `WassersteinFidelity` |
| S | `ExhaustiveCombiGen(k=4)` |
| R | `UniformRepresentationModel` |
| A | `ObjectiveDrivenCombinatorialSearchAlgorithm` with `WeightedSumPolicy` |

## Visualizations

### Responsibility Weights
Uniform 1/k weights for each selected month.
<iframe src="ex3/responsibility_weights.html" width="100%" height="500" frameborder="0"></iframe>
