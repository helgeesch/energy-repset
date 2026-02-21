# A Unified Framework for Representative Subset Selection in Energy Time Series

---

## Abstract

Time series aggregation (TSA), the process of selecting or constructing a small set of representative periods from a larger dataset, is essential for making computationally expensive energy system models tractable. The field has produced a rich but fragmented array of methods: clustering, mathematical programming, autoencoders, greedy algorithms, and more. Each method is typically presented as a monolithic procedure, making it difficult to compare methods, understand their implicit assumptions, or assemble custom pipelines.

This paper proposes a unified framework that decomposes *any* TSA method into five fundamental components: the **Feature Space** (how periods are represented), the **Objective** (what quality means), the **Selection Space** (what form the output takes), the **Representation Model** (how the selection approximates the whole), and the **Search Algorithm** (how the solution is found). We show that every established methodology is a specific instantiation of this five-component structure, and that the framework enables systematic comparison, exposes trade-offs, and provides an architectural blueprint for modular software.

---

## 1. Introduction

Energy system models (ESMs) are central analytical tools for energy system planning and research. The integration of variable renewable energy sources (VRES), storage technologies, and cross-sectoral coupling demands high temporal resolution, typically hourly or sub-hourly, across a full year or longer. For many ESMs, particularly those used in capacity expansion planning, investment optimization, or market studies, this produces optimization problems that are computationally intractable.

Time series aggregation addresses this by selecting a small number $k$ of representative periods (days, weeks, months, or even years) that preserve the essential characteristics of the full dataset. A well-chosen selection can reduce computation by orders of magnitude while maintaining the fidelity of model results.

The challenge is that "representativeness" is not a single, well-defined concept. At a fundamental level, the modeler must decide *what* to represent: the statistical properties of the *input data* (e.g., weather patterns, demand profiles) or the *outcomes* of the downstream model (e.g., socio-economic welfare, system cost, capacity investments). Within these categories, the practical meaning of "representative" varies further depending on the modeling question:

- **Aggregate fidelity**: the selected periods, when weighted appropriately, reproduce the overall statistics (means, distributions, correlations) of the full dataset, so that annualized system cost is captured accurately.
- **State-space coverage**: the selected periods span the diversity of conditions that occur (high wind with high demand, low wind with high solar, peak events, etc.) so that the model encounters all operationally distinct system states.
- **A combination**: cover the breadth of system states *while also* matching aggregate statistics, often under the constraint that all periods carry equal weight.

Different TSA methods make different implicit choices about what to preserve, how to search, and what form the output takes. These choices are rarely made explicit, which makes comparison difficult.

Inspired by unifying efforts in other fields (notably Powell's unified framework for sequential decision problems under uncertainty), we propose a decomposition of the TSA problem into five modular, interchangeable components. Any concrete method is a specific *instantiation* of this structure. The framework does not prescribe a single best method; it provides a common language for describing, comparing, and assembling methods.

The remainder of this paper is organized as follows. Section 2 presents the five-component framework. Section 3 demonstrates how established methods decompose into the framework. Section 4 discusses practical implications and open questions.

---

## 2. The Unified Framework

### 2.1 Overview

Consider a dataset $D = \{d_1, \ldots, d_N\}$ of $N$ time periods, where each $d_i$ is a multivariate time series for period $i$. The variables may include load, wind capacity factors, solar irradiance, temperature, and others, potentially across multiple regions (each region-variable pair is simply an additional dimension of the time series vector). The temporal granularity (days, weeks, months, years) is a problem parameter that depends on the application: one study may select representative days from a year, another representative weeks, and yet another a subset of representative years from a multi-decadal climate dataset.

The goal is to find a *selection* $x$ of $k \ll N$ representative periods (or constructs derived from them) such that some quality measure is optimized. We formalize this as:

$$
x^* \;=\; \underset{x \,\in\, \mathcal{S}}{\arg\min}\; \mathcal{O}\!\bigl(\mathcal{R}(x,\, D)\bigr)
$$

where:

| Symbol | Component | Role |
|--------|-----------|------|
| $\mathcal{F}$ | Feature Space | How periods are represented mathematically |
| $\mathcal{O}$ | Objective | What quality measure is optimized |
| $\mathcal{S}$ | Selection Space | What structural form the output takes |
| $\mathcal{R}$ | Representation Model | How the selection approximates the full dataset |
| $\mathcal{A}$ | Search Algorithm | How the optimal selection is found |

Any concrete TSA method is defined by a specific 5-tuple $(\mathcal{F},\, \mathcal{O},\, \mathcal{S},\, \mathcal{R},\, \mathcal{A})$.

The five components are conceptually independent: each addresses a distinct design decision. In practice, certain combinations are more natural than others, and some methods couple components tightly. Making these couplings explicit is one of the framework's main contributions.

**A note on the role of $\mathcal{R}$ during search.** The formulation above presents the ideal: the objective evaluates the quality of the *representation*, not of the raw selection. In practice, most combinatorial search methods ($\mathcal{A}_\text{comb}$) evaluate candidates by comparing the raw data of the selected periods against the full dataset, effectively bypassing $\mathcal{R}$ during the search and applying it only after the best selection has been identified. This is a pragmatic simplification: computing $\mathcal{R}$ for every candidate in an exhaustive search can be expensive, and for simple representation models like $\mathcal{R}_\text{equal}$ the difference is negligible. Methods that jointly optimize selection and representation, such as $\mathcal{A}_\text{optim}$ (MILP) or $\mathcal{A}_\text{construct}$ (clustering, where the assignment *is* the representation), adhere more closely to the full formulation.

<!-- [FIGURE: Diagram showing the five components and information flow: D → F → (features used by A and O) → x ∈ S → R(x, D) → O evaluates quality. Feedback loop from O to A (search).] -->

The following subsections define each component, its variants, and their trade-offs.

---

### 2.2 Component 1: Feature Space ($\mathcal{F}$) — How We See the Data

Before periods can be compared, grouped, or evaluated, they must be represented as mathematical objects. The feature space $\mathcal{F}$ defines this representation:

$$\mathcal{F}: D \;\to\; \{z_1, \ldots, z_N\}, \quad z_i \in \mathbb{R}^p$$

where $z_i$ is the feature vector for period $i$, and $p$ is the dimensionality of the feature space.

The choice of $\mathcal{F}$ is consequential: it defines what "similar" means. Two periods that are close in one feature space may be distant in another. The feature space shapes how the objective is computed and how the search algorithm operates.

**Variants:**

| Variant | Description |
|---------|-------------|
| $\mathcal{F}_\text{direct}$ | Raw time-series vectors. For a period of $H$ time steps and $V$ variables (where $V$ may include multiple regions): $z_i \in \mathbb{R}^{V \times H}$. Complete but high-dimensional. |
| $\mathcal{F}_\text{stat}$ | Hand-crafted statistical summaries: means, standard deviations, quantiles, ramp rates, correlations, etc. Lower-dimensional and interpretable, but depends on the modeler's choice of features. |
| $\mathcal{F}_\text{latent}$ | Learned low-dimensional representations via PCA, autoencoders, or other dimensionality reduction. Captures complex patterns automatically, including nonlinear structure (autoencoders). |
| $\mathcal{F}_\text{model}$ | Features derived from running a simplified model (e.g., a dispatch model) for each period. Model outputs (generation mix, storage dispatch, marginal prices) are used as features, potentially combined with input data. This makes the feature space "problem-aware": periods are grouped by their *operational impact*, not just their statistical appearance. |

$\mathcal{F}_\text{direct}$ is the default when no explicit feature engineering is performed (e.g., standard k-means on raw hourly data). $\mathcal{F}_\text{stat}$ and $\mathcal{F}_\text{latent}$ trade information for computational efficiency and noise reduction. $\mathcal{F}_\text{model}$ is the most advanced variant, requiring preliminary model runs but offering the strongest link between input representation and downstream model fidelity.

These variants can be composed: for instance, $\mathcal{F}_\text{model}$ features may be passed through an autoencoder to produce a $\mathcal{F}_\text{latent}$ representation that encodes both input patterns and model responses.

#### Two dimensions of temporal complexity reduction

Reducing the temporal complexity of energy system inputs involves two orthogonal dimensions:

1. **Between-period reduction** (representative subset selection): selecting $k$ from $N$ candidate periods. This is the primary focus of the entire framework.
2. **Within-period reduction** (temporal segmentation): adjusting the temporal resolution *inside* each period. This ranges from simple uniform downsampling (e.g., converting 15-minute to hourly data) to domain-informed selective segmentation, where different hours receive different treatment. For example, overnight hours (1am--5am) might be aggregated into a single 4-hour block because little changes operationally, while solar ramp hours (8am--9am) remain at full hourly resolution because the rate of change matters for dispatch.

These two dimensions compose naturally: one can first downsample the temporal resolution and then select representative periods, or apply segmentation after selection. The framework presented here focuses on between-period reduction. Within-period segmentation is an independent technique that can be layered on top. The Python package `tsam` (Kotzur et al., 2018) is one tool that supports both dimensions.

#### Feature normalization and weighting

Two practical considerations arise when constructing $\mathcal{F}$:

**Normalization.** Features often span vastly different scales. Demand variables may be in the range of thousands of MW, while capacity factors lie between 0 and 1. Without normalization, distance-based methods (clustering, Wasserstein distance, diversity metrics) are dominated by variables with the largest magnitude, effectively ignoring the rest. Common strategies include z-score standardization (subtract mean, divide by standard deviation) and min-max scaling. The choice of normalization is a design decision within $\mathcal{F}$ and should be made deliberately.

**Feature weighting.** The modeler may have domain knowledge that certain variables or features matter more for the downstream application than others. For example, if the downstream model is demand-driven, load features should carry more weight than temperature features. Importance weights can be applied at the variable level (scaling entire time series before feature extraction) or at the feature level (scaling individual features in the constructed feature space). Either way, the weighting becomes part of $\mathcal{F}$ and shapes how "similarity" is defined for all downstream components.

#### The curse of dimensionality

The feature dimension $p$ can grow rapidly. With $V$ variables across $R$ regions, the base data contains $V \cdot R$ time series. For $\mathcal{F}_\text{direct}$, each series contributes $H$ features (one per time step), yielding $p = V \cdot R \cdot H$, which can easily reach thousands. For $\mathcal{F}_\text{stat}$, each series produces multiple summary statistics (mean, standard deviation, quantiles, ramp rates, correlations), so $p$ scales with the number of statistics chosen. In either case, the resulting feature space can have hundreds or thousands of dimensions, while the number of candidate periods $N$ may be only 12 (months) to 52 (weeks).

When $p$ is large relative to $N$, distances concentrate: the ratio of the maximum to the minimum pairwise distance approaches 1, and all periods begin to look equally (dis)similar. Clustering degrades, diversity metrics lose discriminative power, and the selection process becomes unreliable.

Practical diagnostics:

- **Feature-to-sample ratio.** Compare $p$ to $N$. If $p \gg N$, dimensionality reduction is strongly advisable.
- **PCA explained variance.** Inspect the cumulative explained variance curve. If a few principal components capture most of the variance (e.g., 3 components explain 90%), the effective dimensionality is manageable.
- **Use $\mathcal{F}_\text{latent}$** (PCA, autoencoders) to project the feature space down to a tractable number of dimensions before running the search.

---

### 2.3 Component 2: Objective ($\mathcal{O}$) — What We Want to Preserve

The objective defines the quality measure that the selection should optimize. This is the most consequential choice in the framework, as it determines what "representative" means for a given application.

$$\mathcal{O}: \mathcal{R}(x, D) \;\to\; \mathbb{R} \quad (\text{or } \mathbb{R}^m \text{ in the multi-objective case})$$

At the highest level, objectives fall into two categories:

| Variant | Description |
|---------|-------------|
| $\mathcal{O}_\text{stat}$ | **Statistical fidelity.** Preserve statistical properties of the input data. This is a *proxy* for the true goal. |
| $\mathcal{O}_\text{model}$ | **Model outcome fidelity.** Preserve the results of the downstream optimization model (total cost, capacity mix, emissions). This is the *true goal*, but typically requires running the model during the selection process. |

Most practical methods use $\mathcal{O}_\text{stat}$ because evaluating $\mathcal{O}_\text{model}$ during the selection process is computationally expensive. The fundamental challenge of TSA is that the relationship between statistical fidelity and model outcome fidelity is complex and nonlinear: low statistical error does not guarantee accurate model results.

#### Statistical objectives in practice

Statistical fidelity is not a single concept. It decomposes into three distinct dimensions, each capturing a different aspect of how well the selection preserves the original data.

**1. Marginal distribution fidelity.** The weighted selection should reproduce the overall distribution of each variable independently: annual means, load duration curves, quantile structures. This is the most commonly targeted fidelity dimension. When the downstream model question concerns aggregate outcomes (annualized system cost, total generation mix), marginal distributions are the primary concern. Metrics include:

- *Wasserstein distance*: how far apart are the full and selected marginal distributions?
- *Duration curve NRMSE*: how well does the weighted selection reconstruct the sorted load/generation profiles?
- *Mean preservation*: does the weighted selection reproduce the annual mean of each variable?

**2. Temporal pattern fidelity.** Energy systems are sensitive not just to *what values* occur, but to *when* and *how fast* things change. Temporal pattern fidelity captures intra-period dynamics: diurnal shapes (solar noon peaks, evening demand ramps), ramp rates (the rate of change between consecutive time steps), and autocorrelation structure. A selection that matches the marginal distribution perfectly can still fail if it misses the characteristic temporal shapes, for instance by selecting periods with flat profiles when the full dataset contains sharp morning ramps. Metrics include:

- *Diurnal profile MSE*: does the selection reproduce the mean hourly shape of each variable?
- *DTW distance*: how similar are the temporal shapes (allowing for slight time shifts)?
- *Ramp-rate statistics*: does the selection preserve the distribution of hour-to-hour changes?

**3. Cross-variable dependency fidelity.** Variables in energy systems do not evolve independently. Wind and solar output are often negatively correlated; demand tends to peak when solar generation ramps down; price spikes coincide with low VRES availability. Preserving these relationships is critical for models that involve cross-sectoral coupling, storage dispatch, or market dynamics. This dimension covers:

- *Static correlations*: does the selection preserve the overall correlation matrix across variables (Frobenius norm of the correlation matrix difference)?
- *Joint temporal dynamics*: does the selection preserve co-movement patterns, such as solar generation ramping up while demand ramps down, or wind output dropping during peak price hours? This goes beyond static correlations to capture the temporal *co-evolution* of variables. One natural measure is the correlation matrix of first-differences (ramp correlations), which quantifies how the rates of change across variables relate to each other.

These three dimensions can be addressed both in **$\mathcal{F}$** (by including ramp-rate statistics, cross-correlations, or derivative-based features in the feature space) and in **$\mathcal{O}$** (by including score components that explicitly measure each fidelity type). In practice, a well-designed selection pipeline addresses all three, either through the choice of features, the choice of objective components, or both.

**State-space coverage.** Complementary to fidelity, the selection should span the diversity of conditions that occur in the full dataset, capturing distinct system states such as "high wind + low demand," "low VRES + peak demand," or "shoulder season with storage cycling." This matters most when the model question concerns system adequacy, resilience, or identifying binding constraints. Metrics include:

- *Diversity* (mean pairwise distance in feature space): are the selected periods sufficiently different from each other?
- *Coverage balance* (uniformity of representation responsibilities): does each selected period "cover" a roughly equal portion of the full dataset?
- *Centroid balance* (distance from selection centroid to global center): does the selection avoid systematic bias toward one region of the feature space?

Diversity and coverage metrics evaluate properties of the selection *itself* in feature space, rather than how well the representation matches the full dataset. They complement fidelity metrics by ensuring the selection is well-spread and balanced.

**Combined objectives.** In many applications, both statistical fidelity and state-space coverage matter simultaneously. A common scenario is the requirement that the selected periods carry **equal weight** (see $\mathcal{R}_\text{equal}$ in Section 2.5), meaning the selection must be intrinsically representative and cannot rely on non-uniform weights to correct for bias. The selection must then:

1. land close to the center of the data distribution (fidelity), *and*
2. span a broad range of system states (coverage/diversity).

These goals are in natural tension: optimizing for fidelity tends to select "average" periods, while optimizing for coverage tends to select "extreme" or "boundary" periods. **Multi-objective optimization** resolves this tension by computing the trade-off frontier (Pareto front) explicitly, letting the modeler choose a preferred balance. This provides a systematic alternative to multi-stage hybrid approaches (see Section 2.6, $\mathcal{A}_\text{hybrid}$).

---

### 2.4 Component 3: Selection Space ($\mathcal{S}$) — What We Are Picking

The selection space defines the structural form of the output, i.e. what kind of object $x$ is.

| Variant | Description |
|---------|-------------|
| $\mathcal{S}_\text{subset}$ | **Historical subset.** $x \subset \{1, \ldots, N\}$ with $\lvert x \rvert = k$. The output is a set of $k$ actual periods from the original data. |
| $\mathcal{S}_\text{synthetic}$ | **Synthetic archetypes.** $x = \{p_1, \ldots, p_k\}$ where each $p_j \in \mathbb{R}^{V \times H}$ is an artificial period (e.g., a cluster centroid). These may not correspond to any historical period. |
| $\mathcal{S}_\text{chrono}$ | **Chronological segments.** $x = \{(t_1, l_1), \ldots, (t_k, l_k)\}$ where $(t_j, l_j)$ defines a contiguous segment of variable length $l_j$ starting at time $t_j$. Segments collectively cover the full timeline. |

$\mathcal{S}_\text{subset}$ is the most common choice because it guarantees that each representative period is a physically realistic, historical pattern. It is the natural output of k-medoids clustering, combinatorial search, and greedy selection methods.

$\mathcal{S}_\text{synthetic}$ arises from methods that construct artificial representatives, such as k-means clustering, where centroids are computed as averages over cluster members. When clustering is performed in $\mathcal{F}_\text{direct}$ (raw time-series space), centroids are themselves time series and can be used directly as synthetic periods, though they may produce physically unrealistic profiles (e.g., smoothed-out peaks). When clustering is performed in a reduced feature space ($\mathcal{F}_\text{stat}$ or $\mathcal{F}_\text{latent}$), the centroid exists in feature space, not in time-series space. Recovering a synthetic time series then requires an inverse mapping, for instance a decoder in the case of $\mathcal{F}_\text{latent}$ (autoencoders). For $\mathcal{F}_\text{stat}$, no natural inverse exists, which is one reason k-medoids ($\mathcal{S}_\text{subset}$) is generally preferred over k-means ($\mathcal{S}_\text{synthetic}$) when working with engineered features.

$\mathcal{S}_\text{chrono}$ preserves the chronological ordering of the original data, which is critical for models with long-duration storage or seasonal dynamics. It is the output of Chronological Time-Period Clustering (CTPC) and related methods.

The choice of $\mathcal{S}$ interacts with other components: $\mathcal{S}_\text{subset}$ pairs naturally with $\mathcal{A}_\text{comb}$ (combinatorial search) and $\mathcal{A}_\text{construct}$ (clustering with medoid selection), while $\mathcal{S}_\text{chrono}$ requires specialized constructive algorithms that enforce contiguity.

---

### 2.5 Component 4: Representation Model ($\mathcal{R}$) — How the Selection Represents the Whole

Given a selection $x$ of $k$ representatives, the representation model defines how the full dataset $D$ is approximated for use in the downstream model. The output of $\mathcal{R}$ is the *reduced input* to the downstream model: a set of representative periods and their associated weights or reconstruction rules.

| Variant | Description |
|---------|-------------|
| $\mathcal{R}_\text{equal}$ | **Equal weighting.** Each selected period receives weight $1/k$. No assignment of original periods to representatives is performed. |
| $\mathcal{R}_\text{hard}$ | **Hard assignment.** Each of the $N$ original periods is assigned to a single closest representative. The weight of representative $j$ is proportional to the number of periods assigned to it. |
| $\mathcal{R}_\text{soft}$ | **Blended representation.** Each original period $i$ is approximated as a weighted combination of all $k$ representatives: $d_i \approx \sum_{j=1}^k w_{ij} \cdot x_j$. |

#### $\mathcal{R}_\text{equal}$: Equal weighting

The simplest representation model. Each of the $k$ selected periods is treated identically in the downstream model, with weight $1/k$.

This choice is common in practice when the downstream model cannot accommodate non-uniform period weights, for instance when the model is formulated to run each representative period once and the results are simply averaged. It places the strongest requirements on the selection itself: since weights cannot compensate for a biased selection, the $k$ periods must be *intrinsically* representative. The aggregate statistics of the equally-weighted selection must match those of the full dataset, while simultaneously covering the relevant state space.

#### $\mathcal{R}_\text{hard}$: Hard assignment

Each original period is mapped to its single closest representative. The weight of representative $j$ reflects the number of original periods it represents:

$$w_j \;=\; \frac{|\{i : r(i) = j\}|}{N}, \quad \text{where } r(i) = \underset{j \in x}{\arg\min}\; \|z_i - z_j\|$$

This is the standard output of clustering methods. The downstream model runs $k$ periods, each weighted by $w_j$, so that the weighted aggregate approximates the full-year result.

The assignment function $r(\cdot)$ and distance metric can vary: Euclidean distance on features, DTW distance on raw series, RBF kernel similarity, or PCA-based k-medoids assignment. The choice of assignment method is a sub-decision within $\mathcal{R}_\text{hard}$.

Variants also exist where the weights are not simply cluster sizes but are **optimized** (e.g., via MILP) to minimize some reconstruction error, decoupling the weight computation from the cluster assignment.

#### $\mathcal{R}_\text{soft}$: Blended representation

Each original period $i$ is approximated as a weighted combination of *all* $k$ representatives:

$$d_i \;\approx\; \sum_{j=1}^k w_{ij} \cdot x_j$$

with constraints on the weight vectors $w_i = (w_{i1}, \ldots, w_{ik})$, typically:

- $w_{ij} \geq 0$ (non-negativity), and optionally
- $\sum_j w_{ij} = 1$ (convex combination) or $\sum_j w_{ij}$ unconstrained (conic combination).

The blended representation provides a much richer approximation: rather than collapsing each original period to a single representative, it reconstructs each one from the full basis of representatives. This is the key idea in hull clustering, where the selected periods form the vertices of a polytope that spans the data.

However, $\mathcal{R}_\text{soft}$ requires the downstream model to handle *blended inputs*: the time-series parameters (load, VRES profiles) for each modeled period are themselves weighted sums of the representative profiles. This requires a different ESM formulation than the standard weighted-period approach.

#### Duration scaling

When periods have unequal durations (for instance, calendar months ranging from 28 to 31 days), the raw responsibility weights produced by any $\mathcal{R}$ variant should be adjusted to reflect the actual time span each representative covers. Without this adjustment, a representative month of 28 days and one of 31 days would receive the same weight despite covering different fractions of the year.

Duration scaling is not a separate representation model but a practical post-processing refinement applicable to $\mathcal{R}_\text{equal}$, $\mathcal{R}_\text{hard}$, and $\mathcal{R}_\text{soft}$ alike. The adjustment is straightforward: each weight $w_j$ is multiplied by the duration $l_j$ of period $j$, and the result is renormalized so that the weights sum to the total time horizon. For $\mathcal{R}_\text{equal}$ with monthly periods, this means a 31-day month receives slightly more weight than a 28-day month, ensuring that the weighted reconstruction accounts for the correct number of hours per period.

---

### 2.6 Component 5: Search Algorithm ($\mathcal{A}$) — How We Find the Solution

The search algorithm is the computational procedure that finds $x^*$ (or an approximation of it). Different algorithms impose different requirements on the other components and exhibit different computational trade-offs.

| Variant | Description |
|---------|-------------|
| $\mathcal{A}_\text{comb}$ | **Combinatorial search.** Enumerate or sample candidate selections from $\mathcal{S}$, evaluate each via $\mathcal{O}$, select the best. |
| $\mathcal{A}_\text{construct}$ | **Constructive algorithms.** Build the solution incrementally: clustering (k-means, k-medoids, hierarchical) or greedy selection (forward selection, hull vertex identification). |
| $\mathcal{A}_\text{optim}$ | **Mathematical programming.** Formulate the selection as an optimization problem (typically MILP) and solve it with a general-purpose solver. |
| $\mathcal{A}_\text{hybrid}$ | **Multi-stage or composite.** Combine multiple algorithms or objectives, e.g., first select "typical" periods via clustering, then add "extreme" periods via optimization. |

#### $\mathcal{A}_\text{comb}$: Combinatorial search

The most flexible approach. Candidate selections are generated (exhaustively or via metaheuristics such as genetic algorithms, simulated annealing, or random sampling) and evaluated against $\mathcal{O}$. This decouples the search from the objective: *any* $\mathcal{O}$ can be used, including multi-objective formulations.

The main limitation is scalability. The number of possible $k$-subsets from $N$ periods is $\binom{N}{k}$, which grows combinatorially. Exhaustive enumeration is feasible only for small problems (e.g., $\binom{52}{8} \approx 7.5 \times 10^8$ is already impractical without further constraints). Metaheuristics can handle larger problems but provide no optimality guarantees.

A key advantage of $\mathcal{A}_\text{comb}$ is its compatibility with multi-objective optimization: by evaluating each candidate on multiple objectives, it naturally produces Pareto fronts, enabling the modeler to inspect trade-offs explicitly.

**Selection policies.** When the objective is multi-dimensional ($\mathcal{O} \to \mathbb{R}^m$), the combinatorial search produces a table of $m$ scores per candidate. A *selection policy* resolves this into a single winner. Common strategies include weighted-sum aggregation (simple but requires choosing weights a priori), utopia-distance methods (select the Pareto-optimal point closest to the ideal), and max-min fairness (select the Pareto-optimal point that maximizes the worst-performing objective). The choice of policy is a sub-decision within $\mathcal{A}_\text{comb}$ that can significantly affect which selection is returned, even when the Pareto front is identical.

**Structured candidate generation.** The scalability of $\mathcal{A}_\text{comb}$ can be improved by constraining the search space at the generation stage. Beyond simple group quotas (e.g., "one period per season"), *hierarchical* generation enables evaluation at a finer granularity than the selection unit. For example, features may be computed at daily resolution while the selection operates at the monthly level: each candidate is a set of complete months, but the objective evaluates the daily data within those months. This hierarchical approach, combining group quotas with multi-resolution evaluation, dramatically reduces the combinatorial space while preserving the flexibility of the generate-and-test paradigm.

#### $\mathcal{A}_\text{construct}$: Constructive algorithms

These algorithms build the solution incrementally rather than evaluating complete candidates.

**Clustering algorithms** (k-means, k-medoids, hierarchical agglomerative) iteratively refine an assignment of periods to clusters. The output is a set of cluster representatives (centroids for k-means, medoids for k-medoids) and an implicit hard assignment. The objective is built into the algorithm: k-means minimizes within-cluster sum of squares; k-medoids minimizes within-cluster sum of distances.

Note: k-medoids selects actual data points as representatives ($\mathcal{S}_\text{subset}$), while k-means produces centroids ($\mathcal{S}_\text{synthetic}$). Hierarchical methods can produce either, depending on how representatives are extracted from the dendrogram.

**Greedy algorithms** build the selection one element at a time. At each step, the period that provides the greatest improvement to the objective is added. Hull clustering uses a greedy strategy to identify extreme points (hull vertices) that define the boundary of the data distribution.

Constructive algorithms are typically fast and scalable, but they couple the search algorithm tightly with the objective. You cannot easily swap in a different $\mathcal{O}$ without changing the algorithm itself.

#### $\mathcal{A}_\text{optim}$: Mathematical programming

The selection problem is formulated as a mathematical optimization problem, typically a Mixed-Integer Linear Program (MILP), with binary decision variables $y_i \in \{0, 1\}$ indicating whether period $i$ is selected:

$$\min \;\mathcal{O}(\ldots) \quad \text{s.t.} \quad \sum_{i=1}^N y_i = k, \quad \text{and problem-specific constraints}$$

This approach can provide global optimality guarantees (within solver tolerances) and naturally handles complex constraints. It is the standard approach for duration-curve-based selection and interregional optimization.

Note that structural constraints such as "select at least one period from each season" do not necessarily require mathematical programming. They can also be enforced within $\mathcal{A}_\text{comb}$ by constraining the candidate generation itself, for example by enumerating only combinations that satisfy group quotas.

The limitation is that the objective must be expressible in a form compatible with the solver (linear or quadratic for LP/MILP). Complex, nonlinear objectives or multi-objective formulations may be difficult to encode.

#### $\mathcal{A}_\text{hybrid}$: Multi-stage and composite approaches

Hybrid approaches combine multiple algorithms or objectives in stages. A common pattern is:

1. Select $k_1$ "typical" periods using a clustering or combinatorial method.
2. Select $k_2 = k - k_1$ "extreme" or "critical stress" periods using optimization-based identification (e.g., identifying periods with the highest system stress via slack variables in a preliminary model run).

This is pragmatic and widely used. Alternatively, the same goal (balancing aggregate fidelity with extreme-event coverage) can be pursued through **multi-objective optimization** within a single $\mathcal{A}_\text{comb}$ framework, where both fidelity and coverage are explicit objectives. The multi-objective approach makes trade-offs transparent and lets the modeler retain full control over the balance, rather than committing to a fixed $k_1$/$k_2$ split a priori.

---

## 3. Methods as Framework Instances

The framework's utility lies in its ability to decompose any TSA method into a specific $(\mathcal{F}, \mathcal{O}, \mathcal{S}, \mathcal{R}, \mathcal{A})$ tuple, making implicit choices explicit and enabling direct comparison.

### 3.1 Decomposition Table

The table below classifies established methodologies. Each row is a specific instantiation of the five components.

| Methodology | $\mathcal{F}$ | $\mathcal{O}$ | $\mathcal{S}$ | $\mathcal{R}$ | $\mathcal{A}$ |
|:---|:---|:---|:---|:---|:---|
| **k-means** | $\mathcal{F}_\text{direct}$ or $\mathcal{F}_\text{stat}$ | $\mathcal{O}_\text{stat}$: min. intra-cluster variance | $\mathcal{S}_\text{synthetic}$ (centroids) | $\mathcal{R}_\text{hard}$ (cluster size) | $\mathcal{A}_\text{construct}$ (clustering) |
| **k-medoids (PAM)** | $\mathcal{F}_\text{direct}$ or $\mathcal{F}_\text{stat}$ | $\mathcal{O}_\text{stat}$: min. intra-cluster distance | $\mathcal{S}_\text{subset}$ (medoids) | $\mathcal{R}_\text{hard}$ (cluster size) | $\mathcal{A}_\text{construct}$ (clustering) |
| **Hierarchical clustering** | $\mathcal{F}_\text{direct}$ or $\mathcal{F}_\text{stat}$ | $\mathcal{O}_\text{stat}$: linkage criterion | $\mathcal{S}_\text{subset}$ or $\mathcal{S}_\text{synthetic}$ | $\mathcal{R}_\text{hard}$ (cluster size) | $\mathcal{A}_\text{construct}$ (clustering) |
| **Duration curve MILP** | $\mathcal{F}_\text{direct}$ | $\mathcal{O}_\text{stat}$: min. duration curve NRMSE | $\mathcal{S}_\text{subset}$ | $\mathcal{R}_\text{hard}$ (optimized weights) | $\mathcal{A}_\text{optim}$ (MILP) |
| **Interregional MILP (NREL)** | $\mathcal{F}_\text{direct}$ | $\mathcal{O}_\text{stat}$: min. regional errors | $\mathcal{S}_\text{subset}$ | $\mathcal{R}_\text{hard}$ (optimized weights) | $\mathcal{A}_\text{optim}$ (MILP) |
| **Autoencoder (inputs only)** | $\mathcal{F}_\text{latent}$ | $\mathcal{O}_\text{stat}$: min. latent distance | $\mathcal{S}_\text{subset}$ (medoids) | $\mathcal{R}_\text{hard}$ (cluster size) | $\mathcal{A}_\text{construct}$ (clustering) |
| **Autoencoder (inputs + outputs)** | $\mathcal{F}_\text{model}$ + $\mathcal{F}_\text{latent}$ | $\mathcal{O}_\text{stat}$: min. latent distance | $\mathcal{S}_\text{subset}$ (medoids) | $\mathcal{R}_\text{hard}$ (cluster size) | $\mathcal{A}_\text{construct}$ (clustering) |
| **DTW-based clustering** | $\mathcal{F}_\text{direct}$ (DTW metric) | $\mathcal{O}_\text{stat}$: min. DTW intra-cluster | $\mathcal{S}_\text{subset}$ or $\mathcal{S}_\text{synthetic}$ | $\mathcal{R}_\text{hard}$ (cluster size) | $\mathcal{A}_\text{construct}$ (clustering) |
| **Snippet algorithm** | $\mathcal{F}_\text{direct}$ (subsequences) | $\mathcal{O}_\text{stat}$: min. subsequence distance | $\mathcal{S}_\text{subset}$ | $\mathcal{R}_\text{hard}$ | $\mathcal{A}_\text{construct}$ (greedy) |
| **CTPC** | $\mathcal{F}_\text{direct}$ | $\mathcal{O}_\text{stat}$: min. intra-cluster distance | $\mathcal{S}_\text{chrono}$ | $\mathcal{R}_\text{hard}$ (implicit) | $\mathcal{A}_\text{construct}$ (hierarchical + contiguity) |
| **Hull clustering (blended)** | $\mathcal{F}_\text{direct}$ or $\mathcal{F}_\text{stat}$ | $\mathcal{O}_\text{stat}$: min. projection error | $\mathcal{S}_\text{subset}$ (hull vertices) | $\mathcal{R}_\text{soft}$ (blended) | $\mathcal{A}_\text{construct}$ (greedy) |
| **Multi-objective Pareto** | $\mathcal{F}_\text{stat}$ | $\mathcal{O}_\text{stat}$: multi-objective (Pareto) | $\mathcal{S}_\text{subset}$ | $\mathcal{R}_\text{equal}$ or $\mathcal{R}_\text{hard}$ | $\mathcal{A}_\text{comb}$ (exhaustive / GA) |
| **Extreme event (slack vars)** | $\mathcal{F}_\text{model}$ | $\mathcal{O}_\text{model}$: max. system stress | $\mathcal{S}_\text{subset}$ | $\mathcal{R}_\text{hard}$ | $\mathcal{A}_\text{hybrid}$ (model-in-loop) |
| **Hybrid (typical + extreme)** | mixed | $\mathcal{O}_\text{stat}$ + $\mathcal{O}_\text{model}$ | $\mathcal{S}_\text{subset}$ | $\mathcal{R}_\text{hard}$ | $\mathcal{A}_\text{hybrid}$ |

### 3.2 Observations

Several insights emerge from the decomposition:

1. **Most methods differ in only one or two components.** Moving from k-medoids to autoencoder-based selection changes only $\mathcal{F}$ (from statistical to latent/model-informed). Moving from k-medoids to hull clustering changes $\mathcal{R}$ (from hard to soft) and $\mathcal{A}$ (from centroid-based to greedy). The other components remain the same. This makes trade-offs explicit and isolable.

2. **$\mathcal{R}_\text{soft}$ is uncommon in the current literature.** Almost all established methods use hard assignment or equal weighting. Blended representation (hull clustering) is a recent innovation that requires changes to the downstream model formulation.

3. **$\mathcal{O}_\text{model}$ is the frontier.** Most methods operate entirely on statistical objectives. Direct optimization for model outcome fidelity requires model-in-the-loop methods (extreme event identification via slack variables, autoencoder with model outputs), which are computationally expensive but represent the state of the art.

4. **Multi-objective optimization offers an alternative to hybrid approaches.** The hybrid approach (typical + extreme) is pragmatic and widely used for balancing aggregate fidelity with extreme-event coverage. Multi-objective formulations provide an alternative: they compute the full trade-off frontier and let the modeler choose, rather than committing to a fixed split.

5. **The choice of $\mathcal{A}$ often dominates the method's identity.** Methods are typically named after their search algorithm (k-means, MILP, genetic algorithm), but the other components, particularly $\mathcal{F}$ and $\mathcal{O}$, often have a greater impact on the quality of the result. The framework redirects attention from *how* the search is conducted to *what* is being optimized and *how* quality is measured.

---

## 4. Discussion

### 4.1 Component Interactions and Coupling

While the five components are conceptually independent, certain combinations are tightly coupled in practice:

- **$\mathcal{A}_\text{construct}$ couples $\mathcal{A}$ with $\mathcal{O}$**: clustering algorithms have their objective function built in (e.g., k-means minimizes within-cluster variance). You cannot freely swap $\mathcal{O}$ without changing $\mathcal{A}$.
- **$\mathcal{R}_\text{soft}$ couples $\mathcal{R}$ with the downstream model**: the ESM must be formulated to accept blended inputs, which is a non-trivial modeling change.
- **$\mathcal{F}_\text{model}$ couples $\mathcal{F}$ with a preliminary model run**: model-informed features require access to a simplified ESM, creating a dependency between the feature engineering pipeline and the modeling workflow.

The framework makes these couplings visible, helping practitioners understand which components they can modify independently and which require coordinated changes.

### 4.2 Practical Guidance

The choice of components should be guided by the modeling question:

- **If aggregate cost accuracy is the priority** and non-uniform period weights are acceptable: $\mathcal{O}_\text{stat}$ with distributional metrics, $\mathcal{R}_\text{hard}$ with optimized weights, $\mathcal{A}_\text{optim}$ (duration curve MILP).
- **If equal-weight periods are required**: $\mathcal{R}_\text{equal}$, combined with a multi-objective $\mathcal{O}$ that balances aggregate fidelity with coverage/diversity, and $\mathcal{A}_\text{comb}$ for explicit trade-off analysis.
- **If the model has significant storage or temporal coupling**: $\mathcal{F}$ should capture temporal dynamics (DTW, ramp features), or $\mathcal{S}_\text{chrono}$ should be used to preserve chronology.
- **If the problem space is very large** (e.g., selecting 10 periods from 365 days): $\mathcal{A}_\text{construct}$ (clustering) or $\mathcal{A}_\text{optim}$ (MILP) for scalability; exhaustive $\mathcal{A}_\text{comb}$ is infeasible.
- **If model outcome fidelity is critical and computational budget allows**: $\mathcal{F}_\text{model}$ with autoencoder, or $\mathcal{A}_\text{hybrid}$ with slack-variable-based extreme identification.

### 4.3 Open Questions

Several aspects of the framework invite further investigation:

1. **Systematic benchmarking.** The framework enables, and calls for, controlled experiments comparing different component combinations on the same datasets and downstream models.
2. **Better proxies for $\mathcal{O}_\text{model}$.** Developing statistical objectives that are stronger predictors of model outcome fidelity, without requiring model-in-the-loop evaluation, remains a major research gap.
3. **Automated component selection.** Can the best $(\mathcal{F}, \mathcal{O}, \mathcal{S}, \mathcal{R}, \mathcal{A})$ tuple be selected automatically based on problem characteristics?
4. **$\mathcal{R}_\text{soft}$ adoption.** Blended representations promise higher fidelity but require ESM reformulation. Quantifying the fidelity gain is important for motivating this effort.

---

## 5. Conclusion

The field of representative period selection for energy time series has produced a rich and growing body of methods, each designed with care for specific use cases. However, these methods are typically presented as monolithic procedures, which obscures their shared structure and makes systematic comparison difficult.

The five-component framework proposed in this paper (Feature Space, Objective, Selection Space, Representation Model, and Search Algorithm) provides a common structure for understanding any TSA method. By decomposing methods into their fundamental choices, the framework:

- makes implicit assumptions explicit,
- enables direct, component-level comparison between methods,
- guides practitioners in assembling custom pipelines suited to their specific modeling questions, and
- provides an architectural blueprint for modular software design.

The framework does not claim that one component combination is universally superior. It provides the vocabulary and structure needed to make informed, transparent choices, moving the field from ad-hoc method selection toward systematic, principled design of representative period selection pipelines.

---

## References

1. Hoffmann et al. (2020). [A Review on Time Series Aggregation Methods for Energy System Models](https://doi.org/10.3390/en13030641). *Energies*, 13(3), 641.
2. Teichgraeber & Brandt (2022). [Time-series aggregation for the optimization of energy systems: Goals, challenges, approaches, and opportunities](https://doi.org/10.1016/j.rser.2021.111984). *Renewable and Sustainable Energy Reviews*, 157, 111984.
3. Nahmmacher et al. (2016). [Carpe diem: A novel approach to select representative days for long-term power system modeling](https://doi.org/10.1016/j.energy.2016.06.081). *Energy*, 112, 430--442.
4. Barbar & Mallapragada (2022). [Representative period selection for power system planning using autoencoder-based dimensionality reduction](https://arxiv.org/abs/2204.13608). *arXiv:2204.13608*.
5. Brown et al. (2025). [An Interregional Optimization Approach for Time Series Aggregation in Continent-Scale Electricity System Models](https://doi.org/10.1016/j.energy.2025.135830). *Energy*, 324, 135830.
6. Anderson et al. (2024). [On the Selection of Intermediate Length Representative Periods for Capacity Expansion](https://arxiv.org/abs/2401.02888). *arXiv:2401.02888*.
7. Neustroev et al. (2025). [Hull Clustering with Blended Representative Periods for Energy System Optimization Models](https://arxiv.org/abs/2508.21641). *arXiv:2508.21641*.
8. Kotzur et al. (2018). [Impact of different time series aggregation methods on optimal energy system design](https://doi.org/10.1016/j.renene.2017.10.017). *Renewable Energy*, 117, 474--487.
9. Bahl et al. (2018). [Typical Periods for Two-Stage Synthesis by Time-Series Aggregation with Bounded Error in Objective Function](https://doi.org/10.3389/fenrg.2017.00035). *Frontiers in Energy Research*, 5, 35.
10. Poncelet et al. (2017). [Selecting Representative Days for Capturing the Implications of Integrating Intermittent Renewables in Generation Expansion Planning Problems](https://doi.org/10.1109/TPWRS.2016.2596803). *IEEE Transactions on Power Systems*, 32(3), 1936--1948.
