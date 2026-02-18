# Example Gallery

Each example script produces interactive Plotly visualizations saved as HTML.
Run the scripts to regenerate outputs in `docs/gallery/`.

---

## [Example 1: Getting Started](ex1.md)

The simplest possible end-to-end workflow. Selects 4 representative months
using a single objective and uniform weights. A minimal "hello world" for
onboarding.

**Components:** `StandardStatsFeatureEngineer` | `WassersteinFidelity` |
`WeightedSumPolicy` | `UniformRepresentationModel` |
`ExhaustiveCombiGen(k=4)`

[View details ->](ex1.md)

---

## [Example 2: Feature Space Exploration](ex2.md)

A comprehensive workflow with monthly slicing, PCA feature engineering, Pareto
selection, and KMedoids cluster-size representation. Showcases the full range of
feature-space diagnostics (scatter plots, correlation heatmap, PCA variance,
feature distributions) and score-component diagnostics (ECDF overlay,
correlation difference, diurnal profiles).

**Components:** `FeaturePipeline` (Stats + PCA) | `WassersteinFidelity` +
`CorrelationFidelity` + `CentroidBalance` | `ParetoMaxMinStrategy` |
`KMedoidsClustersizeRepresentation` | `ExhaustiveCombiGen(k=3)`

[View details ->](ex2.md)

---

## [Example 3: Hierarchical Seasonal Selection](ex3.md)

Selects 4 months (one per season) using day-level features and hierarchical
combination generation. Demonstrates seasonal constraints via
`GroupQuotaHierarchicalCombiGen` and Pareto front visualizations including
parallel coordinates.

**Components:** `StandardStatsFeatureEngineer` (daily) |
`GroupQuotaHierarchicalCombiGen` | `WassersteinFidelity` +
`CorrelationFidelity` | `ParetoMaxMinStrategy` |
`KMedoidsClustersizeRepresentation`

[View details ->](ex3.md)

---

## [Example 4: Comparing Representation Models](ex4.md)

Runs a single search and then applies three different representation models
to the same winning selection, comparing how each distributes responsibility
across the selected months.

**Components:** `FeaturePipeline` (Stats + PCA) | `WassersteinFidelity` +
`CorrelationFidelity` | `WeightedSumPolicy` | `ExhaustiveCombiGen(k=3)` |
`UniformRepresentationModel` + `KMedoidsClustersizeRepresentation` +
`BlendedRepresentationModel`

[View details ->](ex4.md)

---

## [Example 5: Multi-Objective Exploration](ex5.md)

Demonstrates how different score components and selection policies affect the
outcome. Uses 4 objectives and compares `ParetoMaxMinStrategy` vs
`WeightedSumPolicy`.

**Components:** `FeaturePipeline` (Stats + PCA) | `WassersteinFidelity` +
`CorrelationFidelity` + `DurationCurveFidelity` + `DiversityReward` |
`ParetoMaxMinStrategy` vs `WeightedSumPolicy` | `UniformRepresentationModel`
| `ExhaustiveCombiGen(k=3)`

[View details ->](ex5.md)

---

## Running the Examples

```bash
# Install the package
pip install -e .

# Run any example
python examples/ex1_getting_started.py
python examples/ex2_feature_space.py
python examples/ex3_hierarchical_selection.py
python examples/ex4_representation_models.py
python examples/ex5_multi_objective.py
```

Output HTML files are written to `docs/gallery/<example>/` and can be opened
directly in a browser or served via `mkdocs serve`.
