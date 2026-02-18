# Example Gallery

Each example script produces interactive Plotly visualizations saved as HTML.
Run the scripts to regenerate outputs in `docs/gallery/`.

---

## Example 1: Feature Space Exploration

**Script:** `examples/ex1.py`

A comprehensive workflow with monthly slicing, PCA feature engineering, Pareto
selection, and KMedoids cluster-size representation. Showcases feature-space
diagnostics (scatter plots, correlation heatmap, PCA variance) and
score-component diagnostics (ECDF overlay, correlation difference, diurnal
profiles).

**Components:** `FeaturePipeline` (Stats + PCA) | `WassersteinFidelity` +
`CorrelationFidelity` + `CentroidBalance` | `ParetoMaxMinStrategy` |
`KMedoidsClustersizeRepresentation` | `ExhaustiveCombiGen(k=3)`

---

## Example 2: Hierarchical Seasonal Selection

**Script:** `examples/ex2.py`

Selects 4 months (one per season) using day-level features and hierarchical
combination generation. Demonstrates seasonal constraints via
`GroupQuotaHierarchicalCombiGen` and Pareto front visualizations including
parallel coordinates.

**Components:** `StandardStatsFeatureEngineer` (daily) |
`GroupQuotaHierarchicalCombiGen` | `WassersteinFidelity` +
`CorrelationFidelity` | `ParetoMaxMinStrategy` |
`KMedoidsClustersizeRepresentation`

---

## Example 3: Getting Started

**Script:** `examples/ex3_getting_started.py`

The simplest possible end-to-end workflow. Selects 4 representative months
using a single objective and uniform weights. A minimal "hello world" for
onboarding.

**Components:** `StandardStatsFeatureEngineer` | `WassersteinFidelity` |
`WeightedSumPolicy` | `UniformRepresentationModel` |
`ExhaustiveCombiGen(k=4)`

**Visualizations:**

- [Responsibility Weights](ex3/responsibility_weights.html)

---

## Example 4: Comparing Representation Models

**Script:** `examples/ex4_representation_models.py`

Runs a single search and then applies three different representation models
to the same winning selection, comparing how each distributes responsibility
across the selected months.

**Components:** `FeaturePipeline` (Stats + PCA) | `WassersteinFidelity` +
`CorrelationFidelity` | `WeightedSumPolicy` | `ExhaustiveCombiGen(k=3)` |
`UniformRepresentationModel` + `KMedoidsClustersizeRepresentation` +
`BlendedRepresentationModel`

**Visualizations:**

- [Uniform Weights](ex4/responsibility_uniform.html)
- [KMedoids Weights](ex4/responsibility_kmedoids.html)
- [Blended Weights (aggregated)](ex4/responsibility_blended_aggregated.html)
- [Blended Weight Matrix Heatmap](ex4/blended_heatmap.html)

---

## Example 5: Multi-Objective Exploration

**Script:** `examples/ex5_multi_objective.py`

Demonstrates how different score components and selection policies affect the
outcome. Uses 4 objectives and compares `ParetoMaxMinStrategy` vs
`WeightedSumPolicy`.

**Components:** `FeaturePipeline` (Stats + PCA) | `WassersteinFidelity` +
`CorrelationFidelity` + `DurationCurveFidelity` + `DiversityReward` |
`ParetoMaxMinStrategy` vs `WeightedSumPolicy` | `UniformRepresentationModel`
| `ExhaustiveCombiGen(k=3)`

**Visualizations:**

- [Pareto Front (2D)](ex5/pareto_scatter_2d.html)
- [Pareto Scatter Matrix](ex5/pareto_scatter_matrix.html)
- [Score Contributions (Pareto)](ex5/score_contributions_pareto.html)
- [Score Contributions (Weighted Sum)](ex5/score_contributions_weighted_sum.html)
- [Responsibility (Pareto)](ex5/responsibility_pareto.html)
- [Responsibility (Weighted Sum)](ex5/responsibility_weighted_sum.html)
- [Distribution: Load](ex5/histogram_load.html)
- [Distribution: Onshore Wind](ex5/histogram_onwind.html)
- [Distribution: Offshore Wind](ex5/histogram_offwind.html)
- [Distribution: Solar](ex5/histogram_solar.html)
- [Feature Distributions](ex5/feature_distributions.html)

---

## Running the Examples

```bash
# Install the package
pip install -e .

# Run any example
python examples/ex3_getting_started.py
python examples/ex4_representation_models.py
python examples/ex5_multi_objective.py
```

Output HTML files are written to `docs/gallery/<example>/` and can be opened
directly in a browser or served via `mkdocs serve`.
