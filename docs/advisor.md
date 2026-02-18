# Configuration Advisor

This document serves a dual purpose:

1. **Human guide** -- a structured decision tree for choosing energy-repset components.
2. **AI system prompt** -- a self-contained reference an LLM can use to interactively guide users through configuration.

For theory, see [Unified Framework](unified_framework.md). For API details, see [Modules & Components](modules.md).

---

## Component Catalog

### F: Feature Engineering

| Class | Import | Description |
|-------|--------|-------------|
| `StandardStatsFeatureEngineer` | `energy_repset.feature_engineering` | Statistical summaries per slice (mean, std, IQR, quantiles, ramp rates). Z-score normalized. |
| `PCAFeatureEngineer` | `energy_repset.feature_engineering` | PCA dimensionality reduction. Supports variance-threshold or fixed component count. |
| `FeaturePipeline` | `energy_repset.feature_engineering` | Chains multiple engineers sequentially. |

**Typical pipeline:** `StandardStatsFeatureEngineer` -> `PCAFeatureEngineer` (via `FeaturePipeline`).

### O: Score Components

All components implement the `ScoreComponent` protocol with `prepare(context)` and `score(combination)`.

| Class | Direction | What it Measures | Import |
|-------|-----------|------------------|--------|
| `WassersteinFidelity` | min | Marginal distribution similarity (Wasserstein distance, IQR-normalized) | `energy_repset.score_components` |
| `CorrelationFidelity` | min | Cross-variable correlation preservation (Frobenius norm) | `energy_repset.score_components` |
| `DurationCurveFidelity` | min | Duration curve match (quantile-based NRMSE) | `energy_repset.score_components` |
| `NRMSEFidelity` | min | Duration curve match (full interpolation NRMSE) | `energy_repset.score_components` |
| `DiurnalFidelity` | min | Hour-of-day profile preservation (normalized MSE) | `energy_repset.score_components` |
| `DiurnalDTWFidelity` | min | Hour-of-day profile preservation (DTW distance) | `energy_repset.score_components` |
| `DTWFidelity` | min | Full series shape similarity (Dynamic Time Warping) | `energy_repset.score_components` |
| `DiversityReward` | max | Spread in feature space (avg pairwise distance) | `energy_repset.score_components` |
| `CentroidBalance` | min | Feature centroid deviation from global mean | `energy_repset.score_components` |
| `CoverageBalance` | min | Balanced coverage via RBF kernel soft assignment | `energy_repset.score_components` |

Components are bundled into an `ObjectiveSet` (`energy_repset.objectives`) with per-component weights.

### S: Combination Generators

| Class | Import | Description |
|-------|--------|-------------|
| `ExhaustiveCombiGen` | `energy_repset.combi_gens` | All k-of-n combinations. |
| `GroupQuotaCombiGen` | `energy_repset.combi_gens` | Exact quotas per group (e.g., 1 per season). |
| `ExhaustiveHierarchicalCombiGen` | `energy_repset.combi_gens` | Selects parent groups, evaluates on child slices. |
| `GroupQuotaHierarchicalCombiGen` | `energy_repset.combi_gens` | Hierarchical + group quotas. Has `from_slicers_with_seasons()` factory. |

### R: Representation Models

| Class | Import | Description |
|-------|--------|-------------|
| `UniformRepresentationModel` | `energy_repset.representation` | Equal 1/k weights. Returns `dict`. |
| `KMedoidsClustersizeRepresentation` | `energy_repset.representation` | Cluster-proportional weights via k-medoids. Returns `dict`. |
| `BlendedRepresentationModel` | `energy_repset.representation` | Soft assignment (convex combination). Returns weight `DataFrame`. |

### A: Search Algorithms

| Class | Workflow | Import |
|-------|----------|--------|
| `ObjectiveDrivenCombinatorialSearchAlgorithm` | Generate-and-Test | `energy_repset.search_algorithms` |

**Not yet implemented:** Constructive algorithms (Hull Clustering, K-Medoids PAM, Snippet Algorithm), Direct Optimization (MILP).

### Pi: Selection Policies

| Class | Import | Description |
|-------|--------|-------------|
| `WeightedSumPolicy` | `energy_repset.selection_policies` | Scalar aggregation. Supports `normalization='robust_minmax'`. |
| `ParetoMaxMinStrategy` | `energy_repset.selection_policies` | Pareto-optimal solution maximizing worst objective. |
| `ParetoUtopiaPolicy` | `energy_repset.selection_policies` | Pareto-optimal solution closest to utopia point. |

---

## Decision Tree

### Step 1: Understand your data

Ask yourself:

- **Resolution:** Hourly? 15-minute? Daily?
- **Variables:** How many time-series (load, wind, solar, prices, ...)?
- **Horizon:** One year? Multiple years?
- **Candidate count:** How many slices does your `TimeSlicer` produce?
  - 12 months -> C(12,3) = 220 candidates for k=3
  - 52 weeks -> C(52,8) = 752 million candidates for k=8

This determines whether exhaustive search is feasible or you need constrained/hierarchical generation.

### Step 2: Downstream model constraints

Your energy system model may impose constraints on the representation:

| Constraint | Implication |
|------------|-------------|
| Model requires equal-length periods with scalar weights | Use `UniformRepresentationModel` or `KMedoidsClustersizeRepresentation` |
| Model can accept blended inputs (e.g., weighted hourly profiles) | Use `BlendedRepresentationModel` |
| Must cover all seasons | Use `GroupQuotaCombiGen` or `GroupQuotaHierarchicalCombiGen` |
| Must preserve temporal coupling within periods (e.g., multi-day storage) | Prefer weekly/multi-day slicing over monthly |

### Step 3: Computational budget

| Candidate space size | Recommended generator |
|---------------------|----------------------|
| < 10,000 | `ExhaustiveCombiGen` -- evaluate all |
| 10,000 -- 1,000,000 | `GroupQuotaCombiGen` to constrain, or hierarchical generators |
| > 1,000,000 | Hierarchical generators, or future genetic/constructive algorithms |

**Hierarchical trick:** Select at the month level (small combinatorial space) but evaluate on day-level features (high resolution). Use `ExhaustiveHierarchicalCombiGen` or `GroupQuotaHierarchicalCombiGen`.

### Step 4: Quality goals

Choose score components based on what matters for your downstream model:

| Goal | Recommended Components |
|------|----------------------|
| Preserve marginal distributions (load duration curves) | `WassersteinFidelity`, `DurationCurveFidelity`, `NRMSEFidelity` |
| Preserve variable correlations (wind-solar complementarity) | `CorrelationFidelity` |
| Preserve diurnal patterns (solar noon peak, evening ramp) | `DiurnalFidelity`, `DiurnalDTWFidelity` |
| Preserve overall time-series shape | `DTWFidelity` |
| Ensure diverse representatives (avoid redundancy) | `DiversityReward` |
| Balanced coverage of the feature space | `CentroidBalance`, `CoverageBalance` |

**Start simple:** `WassersteinFidelity` + `CorrelationFidelity` covers most needs. Add more components only if you observe specific deficiencies in the results.

### Step 5: Selection policy

| Situation | Recommended Policy |
|-----------|-------------------|
| Single objective or clear priority ranking | `WeightedSumPolicy` (default) |
| Multiple objectives, want balanced trade-off | `WeightedSumPolicy(normalization='robust_minmax')` |
| Multiple objectives, want to avoid worst-case failure | `ParetoMaxMinStrategy` |
| Multiple objectives, want closest to ideal | `ParetoUtopiaPolicy` |

---

## Common Configurations

All examples below assume `import energy_repset as rep`.

### Minimal: single-objective monthly selection

```python
import energy_repset as rep

context = rep.ProblemContext(df_raw=df_raw, slicer=rep.TimeSlicer(unit="month"))
workflow = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.ObjectiveDrivenCombinatorialSearchAlgorithm(
        rep.ObjectiveSet({'wass': (1.0, rep.WassersteinFidelity())}),
        rep.WeightedSumPolicy(),
        rep.ExhaustiveCombiGen(k=4),
    ),
    representation_model=rep.UniformRepresentationModel(),
)
result = rep.RepSetExperiment(context, workflow).run()
```

### Multi-objective with PCA features

```python
feature_pipeline = rep.FeaturePipeline(engineers={
    'stats': rep.StandardStatsFeatureEngineer(),
    'pca': rep.PCAFeatureEngineer(),
})

objective_set = rep.ObjectiveSet({
    'wasserstein': (1.0, rep.WassersteinFidelity()),
    'correlation': (1.0, rep.CorrelationFidelity()),
    'diversity':   (0.5, rep.DiversityReward()),
})

workflow = rep.Workflow(
    feature_engineer=feature_pipeline,
    search_algorithm=rep.ObjectiveDrivenCombinatorialSearchAlgorithm(
        objective_set, rep.ParetoMaxMinStrategy(), rep.ExhaustiveCombiGen(k=3),
    ),
    representation_model=rep.KMedoidsClustersizeRepresentation(),
)
result = rep.RepSetExperiment(context, workflow).run()
```

### Seasonal constraints with hierarchical search

```python
child_slicer = rep.TimeSlicer(unit="day")
context = rep.ProblemContext(df_raw=df_raw, slicer=child_slicer)

combi_gen = rep.GroupQuotaHierarchicalCombiGen.from_slicers_with_seasons(
    parent_k=4,
    dt_index=df_raw.index,
    child_slicer=child_slicer,
    group_quota={'winter': 1, 'spring': 1, 'summer': 1, 'fall': 1},
)

workflow = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.ObjectiveDrivenCombinatorialSearchAlgorithm(
        objective_set, rep.WeightedSumPolicy(), combi_gen,
    ),
    representation_model=rep.KMedoidsClustersizeRepresentation(),
)
result = rep.RepSetExperiment(context, workflow).run()
```

### Blended (soft) representation

```python
workflow = rep.Workflow(
    feature_engineer=feature_pipeline,
    search_algorithm=search_algorithm,
    representation_model=rep.BlendedRepresentationModel(blend_type='convex'),
)
result = rep.RepSetExperiment(context, workflow).run()
# result.weights is a DataFrame (not a dict) for blended models
```

---

## Common Pitfalls

1. **Blended weight aggregation:** `BlendedRepresentationModel.weigh()` returns a weight *matrix*. If you sum columns for visualization, normalize the result so weights sum to 1.0 (otherwise bars show raw sums that scale with N).

2. **Combinatorial explosion:** C(52, 8) = 752 million. Always check `combi_gen.count(slices)` before running. Use hierarchical generators or group quotas to reduce the search space.

3. **PCA without stats:** `PCAFeatureEngineer` operates on existing features. It must come *after* `StandardStatsFeatureEngineer` in a `FeaturePipeline`, not as a standalone.

4. **DTW components are slow:** `DTWFidelity` and `DiurnalDTWFidelity` use dynamic time warping which is O(n^2) per pair. Suitable for small candidate sets; consider cheaper alternatives for large searches.

5. **Direction confusion:** Most fidelity components use `direction="min"` (lower is better). `DiversityReward` uses `direction="max"`. The `ObjectiveSet` and selection policies handle direction automatically -- you do not need to negate scores.

6. **Single vs multi-objective:** With a single score component, `WeightedSumPolicy` and `ParetoMaxMinStrategy` produce identical results. Pareto-based policies only add value with 2+ objectives.
