# Example 5: Multi-Objective Exploration

Demonstrates how different score components and selection policies affect the
outcome. Uses 4 objectives (Wasserstein, Correlation, Duration Curve, Diversity)
and compares `ParetoMaxMinStrategy` vs `WeightedSumPolicy`.

**Script:** [`examples/ex5_multi_objective.py`](https://github.com/helgeesch/energy-repset/blob/main/examples/ex5_multi_objective.py)

| Pillar | Component |
|--------|-----------|
| F | `FeaturePipeline` (Stats + PCA) |
| O | `WassersteinFidelity` + `CorrelationFidelity` + `DurationCurveFidelity` + `DiversityReward` |
| S | `ExhaustiveCombiGen(k=3)` |
| R | `UniformRepresentationModel` |
| A | `ObjectiveDrivenCombinatorialSearchAlgorithm` with `ParetoMaxMinStrategy` and `WeightedSumPolicy` |

## Visualizations

### Pareto Front

#### 2D Scatter (Wasserstein vs Correlation)
Trade-off frontier between distribution fidelity and correlation preservation.
<iframe src="ex5/pareto_scatter_2d.html" width="100%" height="500" frameborder="0"></iframe>

#### Scatter Matrix
Pairwise objective-space scatter matrix for all 4 objectives.
<iframe src="ex5/pareto_scatter_matrix.html" width="100%" height="500" frameborder="0"></iframe>

### Score Contributions

#### ParetoMaxMinStrategy
<iframe src="ex5/score_contributions_pareto.html" width="100%" height="500" frameborder="0"></iframe>

#### WeightedSumPolicy
<iframe src="ex5/score_contributions_weighted_sum.html" width="100%" height="500" frameborder="0"></iframe>

### Responsibility Weights

#### ParetoMaxMinStrategy
<iframe src="ex5/responsibility_pareto.html" width="100%" height="500" frameborder="0"></iframe>

#### WeightedSumPolicy
<iframe src="ex5/responsibility_weighted_sum.html" width="100%" height="500" frameborder="0"></iframe>

### Distribution Overlays (Pareto Selection)

#### Load
<iframe src="ex5/histogram_load.html" width="100%" height="500" frameborder="0"></iframe>

#### Onshore Wind
<iframe src="ex5/histogram_onwind.html" width="100%" height="500" frameborder="0"></iframe>

#### Offshore Wind
<iframe src="ex5/histogram_offwind.html" width="100%" height="500" frameborder="0"></iframe>

#### Solar
<iframe src="ex5/histogram_solar.html" width="100%" height="500" frameborder="0"></iframe>

### Feature Distributions
Histograms of all feature columns.
<iframe src="ex5/feature_distributions.html" width="100%" height="500" frameborder="0"></iframe>

### Diurnal Profiles
Hour-of-day profiles comparing full year and Pareto selection.
<iframe src="ex5/diurnal_profiles.html" width="100%" height="500" frameborder="0"></iframe>

### Correlation Difference
Heatmap of the correlation matrix difference (selection minus full).
<iframe src="ex5/correlation_difference.html" width="100%" height="500" frameborder="0"></iframe>
