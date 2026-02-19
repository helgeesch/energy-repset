# Example 6: Constructive Algorithms

Demonstrates the three constructive search algorithms: Hull Clustering, CTPC,
and Snippet. Unlike the Generate-and-Test workflow (examples 1--5), these
algorithms build solutions directly using their own internal objectives. For
background, see [Constructive Algorithms](../constructive_algorithms.md).

**Script:** [`examples/ex6_constructive_algorithms.py`](https://github.com/helgeesch/energy-repset/blob/main/examples/ex6_constructive_algorithms.py)

## Hull Clustering

Greedy forward selection of months that minimize total projection error.
Paired with `BlendedRepresentationModel` for soft-assignment weights.

| Pillar | Component |
|--------|-----------|
| F | `StandardStatsFeatureEngineer` |
| O | Internal: projection error |
| S | Subset (monthly) |
| R | `BlendedRepresentationModel(blend_type='convex')` |
| A | `HullClusteringSearch(k=3, hull_type='convex')` |

### Responsibility Weights
Aggregated blended weights showing each month's share of responsibility.
<iframe src="hull_responsibility.html" width="100%" height="500" frameborder="0"></iframe>

### Feature Space
Selected hull vertices highlighted in feature space (first two statistical features).
<iframe src="hull_feature_scatter.html" width="100%" height="500" frameborder="0"></iframe>

### ECDF Overlay -- Load
<iframe src="hull_ecdf_load.html" width="100%" height="500" frameborder="0"></iframe>

---

## CTPC (Chronological Time-Period Clustering)

Hierarchical agglomerative clustering with a contiguity constraint: only
temporally adjacent months may merge, producing contiguous segments.

| Pillar | Component |
|--------|-----------|
| F | `StandardStatsFeatureEngineer` |
| O | Internal: within-cluster sum of squares (WCSS) |
| S | Chronological (contiguous segments) |
| R | Pre-computed (segment size fractions) |
| A | `CTPCSearch(k=4, linkage='ward')` |

### Responsibility Weights
Weights proportional to the number of months in each contiguous segment.
<iframe src="ctpc_responsibility.html" width="100%" height="500" frameborder="0"></iframe>

### Feature Space
Segment medoids highlighted in feature space.
<iframe src="ctpc_feature_scatter.html" width="100%" height="500" frameborder="0"></iframe>

### ECDF Overlay -- Load
<iframe src="ctpc_ecdf_load.html" width="100%" height="500" frameborder="0"></iframe>

---

## Snippet Algorithm

Greedy p-median selection of 7-day subsequences from daily profiles. Matches
individual days to their nearest candidate snippet, avoiding the problem where
a single unusual day makes an entire week appear unrepresentative.

| Pillar | Component |
|--------|-----------|
| F | `DirectProfileFeatureEngineer` |
| O | Internal: total per-day minimum distance |
| S | Subset (7-day sliding windows, daily slicing) |
| R | Pre-computed (assignment fractions) |
| A | `SnippetSearch(k=4, period_length_days=7, step_days=7)` |

### Responsibility Weights
Fraction of the year's days assigned to each selected subsequence.
<iframe src="snippet_responsibility.html" width="100%" height="500" frameborder="0"></iframe>

### ECDF Overlay -- Load
<iframe src="snippet_ecdf_load.html" width="100%" height="500" frameborder="0"></iframe>
