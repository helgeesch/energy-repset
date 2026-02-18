# Modules & Components

energy-repset decomposes any representative period selection method into five
interchangeable pillars. Each pillar has a protocol (interface) and one or more
concrete implementations. Swapping a single component changes the behavior
without affecting the rest of the pipeline.

## The Five Pillars

```
Raw DataFrame
  -> TimeSlicer (defines candidate periods)
  -> ProblemContext (holds data + metadata)
  -> [F] FeatureEngineer (creates feature vectors per slice)
  -> [A] SearchAlgorithm (finds optimal selection using [O] ObjectiveSet)
  -> [R] RepresentationModel (calculates weights)
  -> RepSetResult (selection, weights, scores)
```

---

## F: Feature Space

Transforms raw time-series slices into comparable feature vectors.

| Implementation | Description |
|----------------|-------------|
| `StandardStatsFeatureEngineer` | Statistical summaries per slice (mean, std, IQR, quantiles, ramp rates). Z-score normalized. |
| `PCAFeatureEngineer` | PCA dimensionality reduction on existing features. Supports variance-threshold or fixed component count. |
| `FeaturePipeline` | Chains multiple engineers sequentially and concatenates their outputs. |

```python
import energy_repset as rep

# Single engineer
feature_engineer = rep.StandardStatsFeatureEngineer()

# Chained pipeline: compute stats, then reduce with PCA
feature_pipeline = rep.FeaturePipeline(engineers={
    'stats': rep.StandardStatsFeatureEngineer(),
    'pca': rep.PCAFeatureEngineer(),
})
```

---

## O: Objective

An `ObjectiveSet` holds one or more weighted `ScoreComponent` instances.
Each component evaluates how well a candidate selection represents the
full dataset along a specific dimension.

| Component | Name | Direction | What it Measures |
|-----------|------|-----------|------------------|
| `WassersteinFidelity` | `wasserstein` | min | Marginal distribution similarity (Wasserstein distance, IQR-normalized) |
| `CorrelationFidelity` | `correlation` | min | Cross-variable correlation preservation (Frobenius norm) |
| `DurationCurveFidelity` | `nrmse_duration_curve` | min | Duration curve match (quantile-based NRMSE) |
| `NRMSEFidelity` | `nrmse` | min | Duration curve match (full interpolation NRMSE) |
| `DiurnalFidelity` | `diurnal` | min | Hour-of-day profile preservation (normalized MSE) |
| `DiurnalDTWFidelity` | `diurnal_dtw` | min | Hour-of-day profile preservation (DTW distance) |
| `DTWFidelity` | `dtw` | min | Full series shape similarity (Dynamic Time Warping) |
| `DiversityReward` | `diversity` | max | Spread of representatives in feature space (avg pairwise distance) |
| `CentroidBalance` | `centroid_balance` | min | Feature centroid deviation from global mean |
| `CoverageBalance` | `coverage_balance` | min | Balanced coverage via RBF kernel soft assignment |

```python
objective_set = rep.ObjectiveSet({
    'wasserstein': (1.0, rep.WassersteinFidelity()),
    'correlation': (1.0, rep.CorrelationFidelity()),
    'diversity':   (0.5, rep.DiversityReward()),
})
```

The weight (first element of each tuple) expresses relative importance.
Components with `direction="min"` are better when smaller; `direction="max"`
are better when larger.

---

## S: Selection Space

A `CombinationGenerator` defines which subsets the search algorithm considers.

| Implementation | Description |
|----------------|-------------|
| `ExhaustiveCombiGen` | All k-of-n combinations. Feasible for small n (e.g., 12 months, k=4 gives 495 candidates). |
| `GroupQuotaCombiGen` | Enforces exact quotas per group (e.g., 1 month per season). |
| `ExhaustiveHierarchicalCombiGen` | Selects parent groups (e.g., months) but evaluates on child slices (e.g., days). |
| `GroupQuotaHierarchicalCombiGen` | Combines hierarchical selection with group quotas. |

```python
# Simple: all 4-of-12 monthly combinations
combi_gen = rep.ExhaustiveCombiGen(k=4)

# Hierarchical with seasonal constraints
combi_gen = rep.GroupQuotaHierarchicalCombiGen.from_slicers_with_seasons(
    parent_k=4,
    dt_index=df_raw.index,
    child_slicer=rep.TimeSlicer(unit="day"),
    group_quota={'winter': 1, 'spring': 1, 'summer': 1, 'fall': 1},
)
```

---

## R: Representation Model

Determines how selected periods represent the full dataset through
responsibility weights.

| Implementation | Description |
|----------------|-------------|
| `UniformRepresentationModel` | Equal 1/k weights. Simplest option. |
| `KMedoidsClustersizeRepresentation` | Weights proportional to cluster sizes from k-medoids hard assignment. |
| `BlendedRepresentationModel` | Soft assignment: each original slice is a convex combination of representatives. Returns a weight *matrix* instead of a weight dict. |

```python
# Equal weights
uniform = rep.UniformRepresentationModel()

# Cluster-proportional weights
kmedoids = rep.KMedoidsClustersizeRepresentation()

# Soft blending (returns a DataFrame, not a dict)
blended = rep.BlendedRepresentationModel(blend_type='convex')
```

---

## A: Search Algorithm

The engine that finds the optimal selection.

| Implementation | Workflow Type | Description |
|----------------|---------------|-------------|
| `ObjectiveDrivenCombinatorialSearchAlgorithm` | Generate-and-Test | Evaluates all candidate combinations and selects the winner via a `SelectionPolicy`. |

### Selection Policies

The policy decides *how* to pick a winner from the scored candidates:

| Policy | Description |
|--------|-------------|
| `WeightedSumPolicy` | Scalar aggregation of scores. Supports `normalization='robust_minmax'` for multi-objective balance. |
| `ParetoMaxMinStrategy` | Selects the Pareto-optimal solution that maximizes its worst-performing objective. |
| `ParetoUtopiaPolicy` | Selects the Pareto-optimal solution closest to the utopia point. |

```python
# Weighted sum (default)
policy = rep.WeightedSumPolicy(normalization='robust_minmax')
search = rep.ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combi_gen)

# Pareto max-min
policy = rep.ParetoMaxMinStrategy()
search = rep.ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combi_gen)
```

---

## Diagnostics

Interactive Plotly visualizations for inspecting results, feature spaces, and
score component behavior. See the [Gallery](gallery/index.md) for rendered
examples.

### Feature Space

| Class | Purpose |
|-------|---------|
| `FeatureSpaceScatter2D` | 2D scatter plot of feature space |
| `FeatureSpaceScatter3D` | 3D scatter plot of feature space |
| `FeatureSpaceScatterMatrix` | Pairwise scatter matrix |
| `PCAVarianceExplained` | Cumulative variance explained by PCA components |
| `FeatureCorrelationHeatmap` | Correlation heatmap between features |
| `FeatureDistributions` | Distribution histograms per feature |

### Results

| Class | Purpose |
|-------|---------|
| `ResponsibilityBars` | Weight distribution across selected representatives |
| `ParetoScatter2D` | 2D objective-space scatter with Pareto front |
| `ParetoScatterMatrix` | Pairwise objective-space scatter matrix |
| `ParetoParallelCoordinates` | Parallel coordinates of Pareto front |
| `ScoreContributionBars` | Per-component score breakdown |

### Score Components

| Class | Purpose |
|-------|---------|
| `DistributionOverlayECDF` | ECDF comparison of full vs selected data |
| `DistributionOverlayHistogram` | Histogram comparison of full vs selected data |
| `CorrelationDifferenceHeatmap` | Correlation matrix difference heatmap |
| `DiurnalProfileOverlay` | Diurnal profile comparison |

---

## Putting It Together

```python
workflow = rep.Workflow(feature_engineer, search_algorithm, representation_model)
experiment = rep.RepSetExperiment(context, workflow)
result = experiment.run()

# result.selection -> tuple of selected slice identifiers
# result.weights   -> dict mapping each selected slice to its weight
# result.scores    -> dict mapping each objective name to its score
```

For a complete walkthrough, see the [Getting Started](getting_started.md) guide.
For the theoretical foundations, see the [Unified Framework](unified_framework.md).
