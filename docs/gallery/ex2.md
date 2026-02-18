# Example 2: Feature Space Exploration

A comprehensive workflow with monthly slicing, PCA feature engineering, Pareto
selection, and KMedoids cluster-size representation. Showcases the full range of
feature-space diagnostics and score-component diagnostics.

**Script:** [`examples/ex2_feature_space.py`](https://github.com/helgeesch/energy-repset/blob/main/examples/ex2_feature_space.py)

| Pillar | Component |
|--------|-----------|
| F | `FeaturePipeline` (Stats + PCA) |
| O | `WassersteinFidelity` + `CorrelationFidelity` + `CentroidBalance` |
| S | `ExhaustiveCombiGen(k=3)` |
| R | `KMedoidsClustersizeRepresentation` |
| A | `ObjectiveDrivenCombinatorialSearchAlgorithm` with `ParetoMaxMinStrategy` |

## Visualizations

### Feature Space

#### PCA Variance Explained
Cumulative variance captured by each principal component.
<iframe src="ex2/output_pca_variance.html" width="100%" height="500" frameborder="0"></iframe>

#### 2D Feature Scatter (PC0 vs PC1)
Months plotted in the first two principal components.
<iframe src="ex2/output_feature_scatter_2d.html" width="100%" height="500" frameborder="0"></iframe>

#### 3D PCA Projection
Interactive 3D view of the first three principal components.
<iframe src="ex2/output_feature_scatter_3d.html" width="100%" height="500" frameborder="0"></iframe>

#### Feature Correlation Heatmap
Pearson correlations between statistical features.
<iframe src="ex2/output_feature_correlation.html" width="100%" height="500" frameborder="0"></iframe>

#### Scatter Matrix (First 4 PCs)
Pairwise scatter plots of the first four principal components.
<iframe src="ex2/output_feature_scatter_matrix.html" width="100%" height="500" frameborder="0"></iframe>

#### Feature Distributions
Histograms of each feature column.
<iframe src="ex2/output_feature_distributions.html" width="100%" height="500" frameborder="0"></iframe>

### Results

#### Feature Space with Selection
Selected months highlighted in the PC0-PC1 plane.
<iframe src="ex2/output_feature_scatter_with_selection.html" width="100%" height="500" frameborder="0"></iframe>

#### Responsibility Weights
Cluster-proportional weights from KMedoids representation.
<iframe src="ex2/output_responsibility_weights.html" width="100%" height="500" frameborder="0"></iframe>

### Score Component Diagnostics

#### ECDF: Load
Empirical CDF comparison of load between full year and selection.
<iframe src="ex2/output_distribution_ecdf_load.html" width="100%" height="500" frameborder="0"></iframe>

#### Correlation Difference
Heatmap of the correlation matrix difference (selection minus full).
<iframe src="ex2/output_correlation_difference.html" width="100%" height="500" frameborder="0"></iframe>

#### Diurnal Profiles
Hour-of-day profiles comparing full year and selection for all variables.
<iframe src="ex2/output_diurnal_profiles.html" width="100%" height="500" frameborder="0"></iframe>
