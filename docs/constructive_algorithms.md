# Constructive Algorithms

Constructive algorithms build representative selections iteratively using their own internal objectives, rather than scoring pre-generated candidate combinations. They implement **Workflow 2** of the [three generalized workflows](workflow.md): the search algorithm does all the heavy lifting, while the `ObjectiveSet` is only used for post-hoc evaluation.

energy-repset provides four constructive algorithms, each grounded in a published methodology:

| Algorithm | Idea | Selection Space | Weights | Reference |
|-----------|------|-----------------|---------|-----------|
| `KMedoidsSearch` | K-medoids partitioning with medoid selection | Subset | Pre-computed (cluster fractions) | Kaufman & Rousseeuw (1990) |
| `HullClusteringSearch` | Farthest-point greedy hull vertex selection | Subset | External (via `RepresentationModel`) | Neustroev et al. (2025) |
| `CTPCSearch` | Contiguity-constrained hierarchical clustering | Chronological segments | Pre-computed (segment fractions) | Pineda & Morales (2018) |
| `SnippetSearch` | Greedy p-median selection of multi-day subsequences | Subset (sliding windows) | Pre-computed (assignment fractions) | Anderson et al. (2024) |

---

## K-Medoids Clustering

### Idea

K-medoids (PAM --- Partitioning Around Medoids) is the most straightforward clustering-based approach to representative period selection. It partitions $N$ periods into $k$ clusters and selects the **medoid** of each cluster --- the actual data point closest to the cluster center --- as the representative.

Unlike k-means, which produces synthetic centroids that may not correspond to any real period, k-medoids always selects actual historical periods. This makes it a natural fit for the subset selection space ($\mathcal{S}_{\text{subset}}$).

### Algorithm

1. Initialize $k$ medoids (using k-medoids++ or random initialization).
2. Assign each period to the nearest medoid.
3. For each cluster, swap the medoid with the member that minimizes within-cluster distance.
4. Repeat steps 2--3 until convergence or `max_iter`.
5. Compute weights as cluster-size fractions: $w_j = n_j / N$.

### Framework Decomposition

| Pillar | Setting |
|--------|---------|
| F | Any feature space (`StandardStatsFeatureEngineer`, `PCAFeatureEngineer`, `DirectProfileFeatureEngineer`) |
| O | Internal: within-cluster sum of squares (WCSS). External `ObjectiveSet` for post-hoc only. |
| S | Subset ($\mathcal{S} \subset$ original periods) |
| R | Pre-computed (cluster-size fractions). The external `RepresentationModel` is skipped. |
| A | Iterative partitioning (k-medoids) |

### Usage

K-medoids pre-computes weights, so no external `RepresentationModel` is needed:

```python
import energy_repset as rep

workflow = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.KMedoidsSearch(k=4, random_state=42),
    representation_model=None,
)
```

Key parameters:

| Parameter | Effect |
|-----------|--------|
| `k` | Number of clusters / representative periods |
| `metric` | Distance metric (default `'euclidean'`) |
| `method` | `'alternate'` (fast, default) or `'pam'` (exact, slower) |
| `init` | Initialization strategy (default `'k-medoids++'`) |

For a hands-on demo, see [Example 7: K-Medoids Clustering](examples/ex6_clustering.ipynb).

---

## Hull Clustering

**Reference:** G. Neustroev, D. A. Tejada-Arango, G. Morales-Espana, M. M. de Weerdt.
"Hull Clustering with Blended Representative Periods for Energy System Optimization Models."
arXiv: [2508.21641](https://arxiv.org/abs/2508.21641), 2025.

### Idea

Hull Clustering treats representative period selection as a **projection problem**. Given $N$ periods in a $p$-dimensional feature space, the goal is to find $k$ "hull vertices" such that every period can be well approximated as a combination of these vertices.

The key insight is geometric: the selected representatives should span the feature space so that no period is far from the convex hull they define. This naturally produces representatives that cover the extremes of the data --- high-demand winter days, peak-solar summer days, calm wind periods --- because these are the vertices needed to enclose the rest.

### Algorithm

Farthest-point greedy forward selection (Algorithm 2 in the paper):

1. **Initialization:** select the period **furthest from the dataset mean** in feature space.
2. For iterations 2 through $k$, compute each remaining period's projection error (hull distance) by solving:

$$
\min_{\mathbf{w}} \| \mathbf{z}_i - \mathbf{Z}_{\mathcal{S}} \mathbf{w} \|^2
$$

with constraints depending on the hull type:

- **Convex** ($\mathbf{w} \geq 0$, $\sum w_j = 1$): each period is a convex combination of the representatives.
- **Conic** ($\mathbf{w} \geq 0$): each period is a non-negative combination (more relaxed).

3. Select the remaining period with the **maximum** projection error (furthest from the current hull).

This farthest-point strategy naturally selects extreme/boundary periods first --- high-demand winter months, peak-solar summer months, etc. --- producing a hull that spans the data well.

### Framework Decomposition

| Pillar | Setting |
|--------|---------|
| F | Any feature space (`StandardStatsFeatureEngineer`, `PCAFeatureEngineer`, `DirectProfileFeatureEngineer`) |
| O | Internal: total projection error. External `ObjectiveSet` for post-hoc evaluation only. |
| S | Subset ($\mathcal{S} \subset$ original periods) |
| R | External --- typically `BlendedRepresentationModel(blend_type='convex')` to compute soft-assignment weights |
| A | Greedy constructive (farthest-point forward selection) |

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

**Reference:** S. Pineda, J. M. Morales.
"Chronological Time-Period Clustering for Optimal Capacity Expansion Planning With Storage."
*IEEE Transactions on Power Systems*, 33(6), 7162--7170, 2018.
DOI: [10.1109/TPWRS.2018.2842093](https://doi.org/10.1109/TPWRS.2018.2842093)

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

CTPC pre-computes weights, so no external `RepresentationModel` is needed:

```python
import energy_repset as rep

workflow = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.CTPCSearch(k=4, linkage='ward'),
    representation_model=None,
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

**Reference:** O. Anderson, N. Yu, K. Oikonomou, D. Wu.
"On the Selection of Intermediate Length Representative Periods for Capacity Expansion."
arXiv: [2401.02888](https://arxiv.org/abs/2401.02888), 2024.

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
    representation_model=None
)
```

Key parameters:

| Parameter | Effect |
|-----------|--------|
| `period_length_days` | Length of each representative subsequence (e.g., 7 for weekly blocks) |
| `step_days` | Stride between consecutive candidates. `step_days=1` gives maximum overlap (most candidates); `step_days=7` gives non-overlapping windows (fewest candidates, fastest). |

**Implementation notes:**

- The original paper formulates the selection as a MILP (mixed-integer linear program). The energy-repset implementation uses a **greedy p-median** heuristic instead, which provides a $(1 - 1/e)$ approximation guarantee and avoids requiring a MILP solver.
- Distances use **squared Euclidean** distance ($\|\mathbf{d}_i - \mathbf{d}_l\|^2$) rather than the paper's Euclidean norm. This preserves the selection and assignment outcomes (monotone transform) but the reported `total_distance` score is on a squared scale.
- Weights are normalized to **sum to 1** ($w_j = n_j / N$) for consistency with the rest of the energy-repset package, rather than the paper's multiplicity convention ($w_j = n_j / L$).

---

## Comparing the Algorithms

Each constructive algorithm makes different trade-offs:

| Aspect | K-Medoids | Hull Clustering | CTPC | Snippet |
|--------|-----------|----------------|------|---------|
| **Best for** | Standard clustering-based selection | Selecting extreme/boundary periods | Contiguous time blocks | Multi-day representative periods |
| **Weights** | Built-in (cluster fractions) | External (soft assignment) | Built-in (segment fractions) | Built-in (assignment fractions) |
| **Temporal structure** | None (any subset) | None (any subset) | Enforced (contiguous segments) | Partial (sliding windows) |
| **Typical slicing** | Monthly or weekly | Monthly or weekly | Any (monthly, weekly, daily) | Daily (required) |
| **Computational cost** | $O(k \cdot N \cdot I)$ iterations | $O(k \cdot N)$ QP solves | $O(N^2)$ agglomerative clustering | $O(k \cdot N \cdot C)$ distance evaluations |
| **Key strength** | Simple, fast, well-understood | Geometric coverage of feature space | Preserves temporal coupling | Day-level matching within multi-day blocks |

For hands-on comparisons, see [Example 6: K-Medoids Clustering](examples/ex6_clustering.ipynb) and [Example 7: Constructive Algorithms](examples/ex7_constructive_algorithms.ipynb).
