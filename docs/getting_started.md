# Getting Started

This guide walks through a minimal end-to-end workflow, explaining each step
and how it maps to the five-pillar framework (**F**, **O**, **S**, **R**, **A**).

By the end, you will have selected 4 representative months from a year of
hourly time-series data, scored them on distribution fidelity, and generated a
simple diagnostic chart.

## Installation

```bash
pip install energy-repset
```

## Imports

All framework classes are available from the top-level namespace. Diagnostics
live one level down:

```python
import pandas as pd
import energy_repset as rep
import energy_repset.diagnostics as diag
```

## Load Data

energy-repset works with any `pandas.DataFrame` where the index is a
`DatetimeIndex` and each column is a variable (e.g., load, wind, solar):

```python
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)
```

## Define the Problem Context

The `ProblemContext` combines the raw data with a `TimeSlicer` that defines
how the time axis is divided into candidate periods. Here, each calendar
month becomes one candidate:

```python
slicer = rep.TimeSlicer(unit="month")
context = rep.ProblemContext(df_raw=df_raw, slicer=slicer)
print(f"Candidate slices: {context.get_unique_slices()}")
# -> 12 monthly periods
```

## Pillar F: Feature Engineering

Feature engineering transforms the raw time-series into a compact
representation that can be compared across candidate periods.
`StandardStatsFeatureEngineer` computes statistical summaries (mean, std,
quantiles, ramp rates) per slice and variable:

```python
feature_engineer = rep.StandardStatsFeatureEngineer()
```

For richer feature spaces, you can chain engineers with a `FeaturePipeline`:

```python
feature_pipeline = rep.FeaturePipeline(engineers={
    'stats': rep.StandardStatsFeatureEngineer(),
    'pca': rep.PCAFeatureEngineer(),
})
```

In this guide we keep it simple and use only the statistical features.

## Pillar O: Objective

The `ObjectiveSet` defines *how* candidate selections are scored. Each entry
maps a name to a `(weight, ScoreComponent)` tuple. Here we use a single
objective: Wasserstein distance between the marginal distributions of the
selection and the full year.

```python
objective_set = rep.ObjectiveSet({
    'wasserstein': (1.0, rep.WassersteinFidelity()),
})
```

Multiple objectives are easy to add:

```python
objective_set = rep.ObjectiveSet({
    'wasserstein': (1.0, rep.WassersteinFidelity()),
    'correlation': (1.0, rep.CorrelationFidelity()),
})
```

## Pillar S: Selection Space

A `CombinationGenerator` defines which subsets are considered.
`ExhaustiveCombiGen` evaluates every possible k-of-n combination:

```python
k = 4
combi_gen = rep.ExhaustiveCombiGen(k=k)
# For 12 months, k=4 -> C(12,4) = 495 candidates
```

## Pillar A: Search Algorithm

The search algorithm orchestrates the evaluation loop. In the generate-and-test
workflow, it generates candidates via the `CombinationGenerator`, scores each
with the `ObjectiveSet`, and picks a winner using the `SelectionPolicy`:

```python
policy = rep.WeightedSumPolicy()
search_algorithm = rep.ObjectiveDrivenCombinatorialSearchAlgorithm(
    objective_set, policy, combi_gen
)
```

## Pillar R: Representation Model

The representation model determines how the selected periods represent the
full year. `UniformRepresentationModel` assigns equal weight to each
selected period:

```python
representation_model = rep.UniformRepresentationModel()
```

## Run the Workflow

Assemble all components into a `Workflow`, wrap it in a `RepSetExperiment`,
and run:

```python
workflow = rep.Workflow(feature_engineer, search_algorithm, representation_model)
experiment = rep.RepSetExperiment(context, workflow)
result = experiment.run()
```

## Inspect Results

The `RepSetResult` contains the selected periods, their weights, and the
objective scores:

```python
print(f"Selected months: {result.selection}")
print(f"Weights: {result.weights}")
print(f"Wasserstein score: {result.scores['wasserstein']:.4f}")
```

## Diagnostic: Responsibility Bars

energy-repset includes interactive Plotly diagnostics. A responsibility bar
chart shows how the total representation weight is distributed across
selected periods:

```python
fig = diag.ResponsibilityBars().plot(result.weights, show_uniform_reference=True)
fig.show()
```

## Full Script

The complete code is available at
[`examples/ex3_getting_started.py`](https://github.com/helgeesch/energy-repset/blob/main/examples/ex3_getting_started.py).

## Next Steps

- **Swap components**: Try `rep.ParetoMaxMinStrategy` instead of `rep.WeightedSumPolicy`,
  or `rep.KMedoidsClustersizeRepresentation` instead of uniform weights.
  See the [Modules & Components](modules.md) page for all available implementations.
- **Add objectives**: Add `rep.CorrelationFidelity`, `rep.DurationCurveFidelity`, or
  `rep.DiversityReward` to the `ObjectiveSet`.
- **Browse examples**: The [Gallery](gallery/index.md) shows more advanced
  configurations with interactive visualizations.
