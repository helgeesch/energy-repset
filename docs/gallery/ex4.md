# Example 4: Comparing Representation Models

Runs a single search to find the best 3-month selection, then applies three
different representation models to the same winning selection: Uniform,
KMedoids cluster-size, and Blended (soft assignment). Illustrates the R
(representation) pillar of the framework.

**Script:** [`examples/ex4_representation_models.py`](https://github.com/helgeesch/energy-repset/blob/main/examples/ex4_representation_models.py)

| Pillar | Component |
|--------|-----------|
| F | `FeaturePipeline` (Stats + PCA) |
| O | `WassersteinFidelity` + `CorrelationFidelity` |
| S | `ExhaustiveCombiGen(k=3)` |
| R | `UniformRepresentationModel` + `KMedoidsClustersizeRepresentation` + `BlendedRepresentationModel` |
| A | `ObjectiveDrivenCombinatorialSearchAlgorithm` with `WeightedSumPolicy` |

## Visualizations

### Responsibility Weights

#### Uniform Weights
Equal 1/k weights for each selected month.
<iframe src="responsibility_uniform.html" width="100%" height="500" frameborder="0"></iframe>

#### KMedoids Cluster-Size Weights
Weights proportional to the number of months assigned to each cluster.
<iframe src="responsibility_kmedoids.html" width="100%" height="500" frameborder="0"></iframe>

#### Blended Weights (Aggregated)
Soft assignment weights aggregated and normalized to sum to 1.0.
<iframe src="responsibility_blended_aggregated.html" width="100%" height="500" frameborder="0"></iframe>

### Blended Weight Matrix
Heatmap of the full weight matrix: rows are all months, columns are representatives.
<iframe src="blended_heatmap.html" width="100%" height="500" frameborder="0"></iframe>

### Feature Space

#### Feature Scatter with Selection
Selected months highlighted in PCA space (PC0 vs PC1).
<iframe src="feature_scatter_selection.html" width="100%" height="500" frameborder="0"></iframe>

### Distribution Fidelity (ECDF)

#### Load
<iframe src="ecdf_load.html" width="100%" height="500" frameborder="0"></iframe>

#### Onshore Wind
<iframe src="ecdf_onwind.html" width="100%" height="500" frameborder="0"></iframe>

#### Offshore Wind
<iframe src="ecdf_offwind.html" width="100%" height="500" frameborder="0"></iframe>

#### Solar
<iframe src="ecdf_solar.html" width="100%" height="500" frameborder="0"></iframe>
