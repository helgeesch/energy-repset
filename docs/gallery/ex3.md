# Example 3: Hierarchical Seasonal Selection

Selects 4 months (one per season) using day-level features and hierarchical
combination generation. Demonstrates seasonal constraints via
`GroupQuotaHierarchicalCombiGen` and Pareto front visualizations.

**Script:** [`examples/ex3_hierarchical_selection.py`](https://github.com/helgeesch/energy-repset/blob/main/examples/ex3_hierarchical_selection.py)

| Pillar | Component |
|--------|-----------|
| F | `StandardStatsFeatureEngineer` (daily features) |
| O | `WassersteinFidelity` + `CorrelationFidelity` |
| S | `GroupQuotaHierarchicalCombiGen` (1 month per season) |
| R | `KMedoidsClustersizeRepresentation` |
| A | `ObjectiveDrivenCombinatorialSearchAlgorithm` with `ParetoMaxMinStrategy` |

## Visualizations

### Results

#### Responsibility Weights
Cluster-proportional weights for the selected daily slices.
<iframe src="ex3/output_responsibility_weights.html" width="100%" height="500" frameborder="0"></iframe>

#### Pareto Front (2D)
Trade-off between Wasserstein distance and correlation fidelity.
<iframe src="ex3/output_pareto_scatter.html" width="100%" height="500" frameborder="0"></iframe>

#### Pareto Parallel Coordinates
All objectives shown as parallel axes.
<iframe src="ex3/output_pareto_parallel.html" width="100%" height="500" frameborder="0"></iframe>

#### Score Contributions
Normalized contributions of each score component.
<iframe src="ex3/output_score_contributions.html" width="100%" height="500" frameborder="0"></iframe>

### Feature Space

#### Feature Scatter with Selection
First two feature columns with selected days highlighted.
<iframe src="ex3/output_feature_scatter.html" width="100%" height="500" frameborder="0"></iframe>

### Distribution Fidelity (ECDF)

#### Load
<iframe src="ex3/output_ecdf_load.html" width="100%" height="500" frameborder="0"></iframe>

#### Onshore Wind
<iframe src="ex3/output_ecdf_onwind.html" width="100%" height="500" frameborder="0"></iframe>

#### Offshore Wind
<iframe src="ex3/output_ecdf_offwind.html" width="100%" height="500" frameborder="0"></iframe>

#### Solar
<iframe src="ex3/output_ecdf_solar.html" width="100%" height="500" frameborder="0"></iframe>
