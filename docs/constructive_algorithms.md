# Constructive Algorithms

Constructive algorithms build representative selections iteratively using their own internal objectives, rather than scoring pre-generated candidate combinations. They implement **Workflow 2** of the [three generalized workflows](workflow.md): the search algorithm does all the heavy lifting, while the `ObjectiveSet` is only used for post-hoc evaluation.

energy-repset provides three constructive algorithms, each grounded in a published methodology:

| Algorithm | Idea | Selection Space | Weights | Reference |
|-----------|------|-----------------|---------|-----------|
| `HullClusteringSearch` | Greedy projection-error minimization | Subset | External (via `RepresentationModel`) | Bahl et al. (2025) |
| `CTPCSearch` | Contiguity-constrained hierarchical clustering | Chronological segments | Pre-computed (segment fractions) | Kotzur et al. (2018) |
| `SnippetSearch` | Greedy p-median selection of multi-day subsequences | Subset (sliding windows) | Pre-computed (assignment fractions) | Teichgraeber & Brandt (2024) |

---

## Hull Clustering

**Reference:** B. Bahl, M. Feldmeier, H. Barber, A. Bardow.
"A Projection-Based Method for Selecting Representative Periods in Energy System Models."
*arXiv:2508.21641*, 2025.

### Idea

Hull Clustering treats representative period selection as a **projection problem**. Given $N$ periods in a $p$-dimensional feature space, the goal is to find $k$ "hull vertices" such that every period can be well approximated as a combination of these vertices.

The key insight is geometric: the selected representatives should span the feature space so that no period is far from the convex hull they define. This naturally produces representatives that cover the extremes of the data --- high-demand winter days, peak-solar summer days, calm wind periods --- because these are the vertices needed to enclose the rest.

### Algorithm

Greedy forward selection. At each of $k$ iterations:

1. For every remaining candidate $c$, tentatively add it to the current selection $\mathcal{S}$.
2. For each period $i$, solve a small projection problem:

$$
\min_{\mathbf{w}} \| \mathbf{z}_i - \mathbf{Z}_{\mathcal{S}} \mathbf{w} \|^2
$$

with constraints depending on the hull type:

- **Convex** ($\mathbf{w} \geq 0$, $\sum w_j = 1$): each period is a convex combination of the representatives.
- **Conic** ($\mathbf{w} \geq 0$): each period is a non-negative combination (more relaxed).

3. Select the candidate that most reduces the total projection error.

### Framework Decomposition

| Pillar | Setting |
|--------|---------|
| F | Any feature space (`StandardStatsFeatureEngineer`, `PCAFeatureEngineer`, `DirectProfileFeatureEngineer`) |
| O | Internal: total projection error. External `ObjectiveSet` for post-hoc evaluation only. |
| S | Subset ($\mathcal{S} \subset$ original periods) |
| R | External --- typically `BlendedRepresentationModel(blend_type='convex')` to compute soft-assignment weights |
| A | Greedy constructive (forward selection) |

### Usage

Hull Clustering leaves `weights=None` in the result, so it is naturally paired with `BlendedRepresentationModel` which computes soft-assignment weights using the same convex projection logic:

```python
import energy_repset as rep

workflow = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.HullClusteringSearch(k=4, hull_type='convex'),
    representation_model=rep.BlendedRepresentationModel(blend_type='convex'),
)
```

---

## CTPC (Chronological Time-Period Clustering)

**Reference:** L. Kotzur, P. Markewitz, M. Robinius, D. Stolten.
"Impact of different time series aggregation methods on optimal energy system design."
*Renewable Energy*, 117, 474-487, 2018.
DOI: [10.1016/j.renene.2017.10.017](https://doi.org/10.1016/j.renene.2017.10.017)

### Idea

CTPC applies **hierarchical agglomerative clustering with a contiguity constraint**: only temporally adjacent periods may merge. This guarantees that the resulting clusters are contiguous time segments --- for example, "January 1--March 15" and "March 16--June 30" rather than scattered individual periods.

This is valuable when the downstream model needs to preserve temporal coupling across periods (e.g., multi-day storage dispatch, seasonal hydro scheduling) or when the user wants to reason about contiguous blocks of time.

### Algorithm

1. Arrange all $N$ time slices in chronological order.
2. Build a tridiagonal connectivity matrix: each slice connects only to its immediate neighbors.
3. Run agglomerative clustering (Ward, complete, average, or single linkage) with this connectivity constraint, stopping at $k$ clusters.
4. Within each cluster, select the **medoid** (period closest to the cluster centroid) as the representative.
5. Compute weights as the fraction of time covered by each segment: $w_j = |\text{segment}_j| / N$.

### Framework Decomposition

| Pillar | Setting |
|--------|---------|
| F | Any feature space (`StandardStatsFeatureEngineer`, `DirectProfileFeatureEngineer`) |
| O | Internal: within-cluster sum of squares (WCSS). External `ObjectiveSet` for post-hoc only. |
| S | Chronological (contiguous time segments) |
| R | Pre-computed (segment size fractions). The external `RepresentationModel` is skipped. |
| A | Hierarchical agglomerative with contiguity constraint |

### Usage

CTPC pre-computes weights, so the `RepresentationModel` in the workflow is skipped. Use `UniformRepresentationModel()` as a placeholder:

```python
import energy_repset as rep

workflow = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.CTPCSearch(k=4, linkage='ward'),
    representation_model=rep.UniformRepresentationModel(),  # placeholder, skipped
)
```

The `linkage` parameter controls how inter-cluster distance is measured during merging:

| Linkage | Behavior |
|---------|----------|
| `'ward'` | Minimizes within-cluster variance (default, usually best) |
| `'complete'` | Maximum distance between any two points in different clusters |
| `'average'` | Mean distance between all pairs across clusters |
| `'single'` | Minimum distance between any two points in different clusters |

---

## Snippet Algorithm

**Reference:** H. Teichgraeber, A. Brandt.
"Time-series aggregation for the optimization of energy systems: Goals, challenges, approaches, and opportunities."
*Renewable and Sustainable Energy Reviews*, 2024.
arXiv: [2401.02888](https://arxiv.org/abs/2401.02888)

### Idea

The Snippet algorithm addresses a specific limitation of period-level selection: when selecting entire weeks (or multi-day blocks), a single anomalous day can make an otherwise typical week appear unrepresentative. Snippet solves this by comparing at the **day level within multi-day windows**.

The algorithm selects $k$ sliding-window subsequences (e.g., 7-day blocks) from the time horizon. The distance from any individual day to a candidate subsequence is the minimum distance to any of that subsequence's constituent daily profiles. This means a candidate week is "close" to a given day if *any* of its 7 days resembles that day --- not just the week as a whole.

### Algorithm

1. Flatten each day's hourly values across all variables into a profile vector $\mathbf{d}_i \in \mathbb{R}^{H \times V}$ (e.g., 24 hours $\times$ 3 variables = 72 dimensions).
2. Generate $C$ sliding-window candidates of length $L$ days with stride $s$:
$$
C = \lfloor (N - L) / s \rfloor + 1
$$
3. Compute a distance matrix $\mathbf{D} \in \mathbb{R}^{N \times C}$ where:
$$
D_{i,j} = \min_{l \in \text{candidate}_j} \| \mathbf{d}_i - \mathbf{d}_l \|^2
$$
4. Greedy p-median selection ($k$ iterations): at each step, pick the candidate that most reduces the total per-day minimum distance.
5. Assign each day to its nearest selected candidate. Weights are the fraction of assigned days: $w_j = n_j / N$.

### Framework Decomposition

| Pillar | Setting |
|--------|---------|
| F | `DirectProfileFeatureEngineer` (raw hourly profiles, flattened) |
| O | Internal: total per-day minimum distance. External `ObjectiveSet` for post-hoc only. |
| S | Subset (sliding-window subsequences) |
| R | Pre-computed (assignment fractions). The external `RepresentationModel` is skipped. |
| A | Greedy p-median |

### Usage

Snippet requires daily slicing and works best with `DirectProfileFeatureEngineer`:

```python
import energy_repset as rep

slicer = rep.TimeSlicer(unit='day')
context = rep.ProblemContext(df_raw=df_raw, slicer=slicer)

workflow = rep.Workflow(
    feature_engineer=rep.DirectProfileFeatureEngineer(),
    search_algorithm=rep.SnippetSearch(k=4, period_length_days=7, step_days=1),
    representation_model=rep.UniformRepresentationModel(),  # placeholder, skipped
)
```

Key parameters:

| Parameter | Effect |
|-----------|--------|
| `period_length_days` | Length of each representative subsequence (e.g., 7 for weekly blocks) |
| `step_days` | Stride between consecutive candidates. `step_days=1` gives maximum overlap (most candidates); `step_days=7` gives non-overlapping windows (fewest candidates, fastest). |

---

## Comparing the Algorithms

Each constructive algorithm makes different trade-offs:

| Aspect | Hull Clustering | CTPC | Snippet |
|--------|----------------|------|---------|
| **Best for** | Selecting extreme/boundary periods | Contiguous time blocks | Multi-day representative periods |
| **Weights** | External (soft assignment) | Built-in (segment fractions) | Built-in (assignment fractions) |
| **Temporal structure** | None (any subset) | Enforced (contiguous segments) | Partial (sliding windows) |
| **Typical slicing** | Monthly or weekly | Any (monthly, weekly, daily) | Daily (required) |
| **Computational cost** | $O(k \cdot N^2)$ QP solves | $O(N^2)$ agglomerative clustering | $O(k \cdot N \cdot C)$ distance evaluations |
| **Key strength** | Geometric coverage of feature space | Preserves temporal coupling | Day-level matching within multi-day blocks |

For a hands-on comparison, see [Example 6: Constructive Algorithms](gallery/ex6.md).
