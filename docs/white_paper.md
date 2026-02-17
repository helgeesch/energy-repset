# A Unified Framework for Representative Subset Selection in Energy Time Series

---

## Abstract

Time series aggregation (TSA) — selecting or constructing a small set of representative periods from a larger dataset — is essential for making computationally expensive energy system models tractable. The field has produced a rich but fragmented array of methods: clustering, mathematical programming, autoencoders, greedy algorithms, and more. Each method is typically presented as a monolithic procedure, making it difficult to compare methods, understand their implicit assumptions, or assemble custom pipelines.

This paper proposes a unified framework that decomposes *any* TSA method into five fundamental components: the **Feature Space** (how periods are represented), the **Objective** (what quality means), the **Selection Space** (what form the output takes), the **Representation Model** (how the selection approximates the whole), and the **Search Algorithm** (how the solution is found). We show that every established methodology is a specific instantiation of this five-component structure, and that the framework enables systematic comparison, exposes trade-offs, and provides an architectural blueprint for modular software.

---

## 1. Introduction

Energy system models (ESMs) are central analytical tools for energy system planning and research. The integration of variable renewable energy sources (VRES), storage technologies, and cross-sectoral coupling demands high temporal resolution — typically hourly or sub-hourly — across a full year or longer. For many ESMs, particularly those used in capacity expansion planning, investment optimization, or market studies, this produces optimization problems that are computationally intractable.

Time series aggregation addresses this by selecting a small number $k$ of representative periods (days, weeks, months, or even years) that preserve the essential characteristics of the full dataset. A well-chosen selection can reduce computation by orders of magnitude while maintaining the fidelity of model results.

The challenge is that "representativeness" is not a single, well-defined concept. At a fundamental level, the modeler must decide *what* to represent: the statistical properties of the *input data* (e.g., weather patterns, demand profiles) or the *outcomes* of the downstream model (e.g., socio-economic welfare, system cost, capacity investments). Within these categories, the practical meaning of "representative" varies further depending on the modeling question:

- **Aggregate fidelity**: the selected periods, when weighted appropriately, reproduce the overall statistics (means, distributions, correlations) of the full dataset — for instance, so that annualized system cost is captured accurately.
- **State-space coverage**: the selected periods span the diversity of conditions that occur — high wind with high demand, low wind with high solar, peak events, etc. — so that the model encounters all operationally distinct system states.
- **A combination**: cover the breadth of system states *while also* matching aggregate statistics — often under the constraint that all periods carry equal weight.

Different TSA methods make different implicit choices about what to preserve, how to search, and what form the output takes. These choices are rarely made explicit, which makes comparison difficult.

Inspired by unifying efforts in other fields — notably Powell's unified framework for sequential decision problems under uncertainty — we propose a decomposition of the TSA problem into five modular, interchangeable components. Any concrete method is a specific *instantiation* of this structure. The framework does not prescribe a single best method; rather, it provides a common language for describing, comparing, and assembling methods.

The remainder of this paper is organized as follows. Section 2 presents the five-component framework. Section 3 demonstrates how established methods decompose into the framework. Section 4 discusses practical implications and open questions.

---

## 2. The Unified Framework

### 2.1 Overview

Consider a dataset $D = \{d_1, \ldots, d_N\}$ of $N$ time periods, where each $d_i$ is a multivariate time series for period $i$. The variables may include load, wind capacity factors, solar irradiance, temperature, and others, potentially across multiple regions — each region-variable pair is simply an additional dimension of the time series vector. The temporal granularity — days, weeks, months, years — is a problem parameter that depends on the application: one study may select representative days from a year, another representative weeks, and yet another a subset of representative years from a multi-decadal climate dataset.

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
| $\mathcal{F}_\text{model}$ | Features derived from running a simplified model (e.g., a dispatch model) for each period. Model outputs — generation mix, storage dispatch, marginal prices — are used as features, potentially combined with input data. This makes the feature space "problem-aware": periods are grouped by their *operational impact*, not just their statistical appearance. |

$\mathcal{F}_\text{direct}$ is the default when no explicit feature engineering is performed (e.g., standard k-means on raw hourly data). $\mathcal{F}_\text{stat}$ and $\mathcal{F}_\text{latent}$ trade information for computational efficiency and noise reduction. $\mathcal{F}_\text{model}$ is the most advanced variant, requiring preliminary model runs but offering the strongest link between input representation and downstream model fidelity.

These variants can be composed: for instance, $\mathcal{F}_\text{model}$ features may be passed through an autoencoder to produce a $\mathcal{F}_\text{latent}$ representation that encodes both input patterns and model responses.

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

In practice, the modeler faces a fundamental choice about *what kind of statistical fidelity matters*:

**Aggregate fidelity.** The selection should reproduce the overall statistics of the full dataset — annual means, marginal distributions, correlation structures, duration curves. This is the right goal when the model question concerns aggregate outcomes (e.g., annualized system cost, total generation mix). Metrics include:

- *Distributional distance* (Wasserstein, duration curve NRMSE): how well does the weighted selection reproduce the marginal distributions?
- *Correlation fidelity* (Frobenius norm of correlation matrix difference): does the selection preserve the dependencies between variables?
- *Diurnal pattern fidelity* (MSE of mean hourly profiles): does the selection reproduce typical intra-day shapes?

**State-space coverage.** The selection should span the diversity of conditions that occur in the full dataset — capturing distinct system states such as "high wind + low demand," "low VRES + peak demand," "shoulder season with storage cycling," etc. This is the right goal when the model question concerns system adequacy, resilience, or the identification of binding constraints. Metrics include:

- *Diversity* (mean pairwise distance in feature space): are the selected periods sufficiently different from each other?
- *Coverage balance* (uniformity of representation responsibilities): does each selected period "cover" a roughly equal portion of the full dataset?

**Combined objectives.** In many applications, both aggregate fidelity and state-space coverage matter simultaneously. A common practical scenario is the requirement that the selected periods carry **equal weight** (see $\mathcal{R}_\text{equal}$ in Section 2.5), which means the selection must be intrinsically representative — it cannot rely on non-uniform weights to correct for bias. In this setting, the selection must:

1. land close to the center of the data distribution (aggregate fidelity), *and*
2. span a broad range of system states (coverage/diversity).

These two goals are in natural tension: optimizing for aggregate fidelity tends to select "average" periods, while optimizing for coverage tends to select "extreme" or "boundary" periods. This tension is resolved through **multi-objective optimization**, where the trade-off frontier (Pareto front) between the objectives is computed explicitly, and the modeler chooses a preferred balance. Multi-objective formulations thus provide a principled alternative to ad-hoc multi-stage approaches (see Section 2.6, $\mathcal{A}_\text{hybrid}$).

---

### 2.4 Component 3: Selection Space ($\mathcal{S}$) — What We Are Picking

The selection space defines the structural form of the output — what kind of object $x$ is.

| Variant | Description |
|---------|-------------|
| $\mathcal{S}_\text{subset}$ | **Historical subset.** $x \subset \{1, \ldots, N\}$ with $\lvert x \rvert = k$. The output is a set of $k$ actual periods from the original data. |
| $\mathcal{S}_\text{synthetic}$ | **Synthetic archetypes.** $x = \{p_1, \ldots, p_k\}$ where each $p_j \in \mathbb{R}^{V \times H}$ is an artificial period (e.g., a cluster centroid). These may not correspond to any historical period. |
| $\mathcal{S}_\text{chrono}$ | **Chronological segments.** $x = \{(t_1, l_1), \ldots, (t_k, l_k)\}$ where $(t_j, l_j)$ defines a contiguous segment of variable length $l_j$ starting at time $t_j$. Segments collectively cover the full timeline. |

$\mathcal{S}_\text{subset}$ is the most common choice because it guarantees that each representative period is a physically realistic, historical pattern. It is the natural output of k-medoids clustering, combinatorial search, and greedy selection methods.

$\mathcal{S}_\text{synthetic}$ arises from methods that construct artificial representatives, such as k-means clustering, where centroids are computed as averages over cluster members. When clustering is performed in $\mathcal{F}_\text{direct}$ (raw time-series space), centroids are themselves time series and can be used directly as synthetic periods — though they may produce physically unrealistic profiles (e.g., smoothed-out peaks). When clustering is performed in a reduced feature space ($\mathcal{F}_\text{stat}$ or $\mathcal{F}_\text{latent}$), the centroid exists in feature space, not in time-series space. Recovering a synthetic time series then requires an inverse mapping — for instance, a decoder in the case of $\mathcal{F}_\text{latent}$ (autoencoders). For $\mathcal{F}_\text{stat}$, no natural inverse exists, which is one reason k-medoids ($\mathcal{S}_\text{subset}$) is generally preferred over k-means ($\mathcal{S}_\text{synthetic}$) when working with engineered features.

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

This choice is common in practice when the downstream model cannot accommodate non-uniform period weights — for instance, when the model is formulated to run each representative period once, and the results are simply averaged. It places the strongest requirements on the selection itself: since weights cannot compensate for a biased selection, the $k$ periods must be *intrinsically* representative. The aggregate statistics of the equally-weighted selection must match those of the full dataset, while simultaneously covering the relevant state space.

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

However, $\mathcal{R}_\text{soft}$ requires the downstream model to handle *blended inputs* — the time-series parameters (load, VRES profiles) for each modeled period are themselves weighted sums of the representative profiles. This requires a different ESM formulation than the standard weighted-period approach.

---

### 2.6 Component 5: Search Algorithm ($\mathcal{A}$) — How We Find the Solution

The search algorithm is the computational procedure that finds $x^*$ (or an approximation of it). Different algorithms impose different requirements on the other components and exhibit different computational trade-offs.

| Variant | Description |
|---------|-------------|
| $\mathcal{A}_\text{comb}$ | **Combinatorial search.** Enumerate or sample candidate selections from $\mathcal{S}$, evaluate each via $\mathcal{O}$, select the best. |
| $\mathcal{A}_\text{construct}$ | **Constructive algorithms.** Build the solution incrementally: clustering (k-means, k-medoids, hierarchical) or greedy selection (forward selection, hull vertex identification). |
| $\mathcal{A}_\text{optim}$ | **Mathematical programming.** Formulate the selection as an optimization problem (typically MILP) and solve it with a general-purpose solver. |
| $\mathcal{A}_\text{hybrid}$ | **Multi-stage or composite.** Combine multiple algorithms or objectives — e.g., first select "typical" periods via clustering, then add "extreme" periods via optimization. |

#### $\mathcal{A}_\text{comb}$: Combinatorial search

The most flexible approach. Candidate selections are generated (exhaustively or via metaheuristics such as genetic algorithms, simulated annealing, or random sampling) and evaluated against $\mathcal{O}$. This decouples the search from the objective: *any* $\mathcal{O}$ can be used, including multi-objective formulations.

The main limitation is scalability. The number of possible $k$-subsets from $N$ periods is $\binom{N}{k}$, which grows combinatorially. Exhaustive enumeration is feasible only for small problems (e.g., $\binom{52}{8} \approx 7.5 \times 10^8$ is already impractical without further constraints). Metaheuristics can handle larger problems but provide no optimality guarantees.

A key advantage of $\mathcal{A}_\text{comb}$ is its compatibility with multi-objective optimization: by evaluating each candidate on multiple objectives, it naturally produces Pareto fronts, enabling the modeler to inspect trade-offs explicitly.

#### $\mathcal{A}_\text{construct}$: Constructive algorithms

These algorithms build the solution incrementally rather than evaluating complete candidates.

**Clustering algorithms** (k-means, k-medoids, hierarchical agglomerative) iteratively refine an assignment of periods to clusters. The output is a set of cluster representatives (centroids for k-means, medoids for k-medoids) and an implicit hard assignment. The objective is built into the algorithm: k-means minimizes within-cluster sum of squares; k-medoids minimizes within-cluster sum of distances.

Note: k-medoids selects actual data points as representatives ($\mathcal{S}_\text{subset}$), while k-means produces centroids ($\mathcal{S}_\text{synthetic}$). Hierarchical methods can produce either, depending on how representatives are extracted from the dendrogram.

**Greedy algorithms** build the selection one element at a time. At each step, the period that provides the greatest improvement to the objective is added. Hull clustering uses a greedy strategy to identify extreme points (hull vertices) that define the boundary of the data distribution.

Constructive algorithms are typically fast and scalable, but they couple the search algorithm tightly with the objective — you cannot easily swap in a different $\mathcal{O}$ without changing the algorithm itself.

#### $\mathcal{A}_\text{optim}$: Mathematical programming

The selection problem is formulated as a mathematical optimization problem, typically a Mixed-Integer Linear Program (MILP), with binary decision variables $y_i \in \{0, 1\}$ indicating whether period $i$ is selected:

$$\min \;\mathcal{O}(\ldots) \quad \text{s.t.} \quad \sum_{i=1}^N y_i = k, \quad \text{and problem-specific constraints}$$

This approach can provide global optimality guarantees (within solver tolerances) and naturally handles complex constraints. It is the standard approach for duration-curve-based selection and interregional optimization.

Note that structural constraints such as "select at least one period from each season" do not necessarily require mathematical programming. They can also be enforced within $\mathcal{A}_\text{comb}$ by constraining the candidate generation itself — for example, by enumerating only combinations that satisfy group quotas.

The limitation is that the objective must be expressible in a form compatible with the solver (linear or quadratic for LP/MILP). Complex, nonlinear objectives or multi-objective formulations may be difficult to encode.

#### $\mathcal{A}_\text{hybrid}$: Multi-stage and composite approaches

Hybrid approaches combine multiple algorithms or objectives in stages. A common pattern is:

1. Select $k_1$ "typical" periods using a clustering or combinatorial method.
2. Select $k_2 = k - k_1$ "extreme" or "critical stress" periods using optimization-based identification (e.g., identifying periods with the highest system stress via slack variables in a preliminary model run).

This is pragmatic and widely used. However, the same goal — balancing aggregate fidelity with extreme-event coverage — can often be achieved through **multi-objective optimization** within a single $\mathcal{A}_\text{comb}$ framework, where both fidelity and coverage are explicit objectives. The advantage of the multi-objective approach is that trade-offs are made transparent and the modeler retains full control over the balance, rather than committing to a fixed $k_1$/$k_2$ split a priori.

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

2. **$\mathcal{R}_\text{soft}$ remains rare.** Almost all established methods use hard assignment or equal weighting. Blended representation (hull clustering) is a recent innovation that requires changes to the downstream model formulation.

3. **$\mathcal{O}_\text{model}$ is the frontier.** Most methods operate entirely on statistical objectives. Direct optimization for model outcome fidelity requires model-in-the-loop methods (extreme event identification via slack variables, autoencoder with model outputs), which are computationally expensive but represent the state of the art.

4. **Multi-objective optimization is underutilized.** The hybrid approach (typical + extreme) is a pragmatic but ad-hoc solution to the tension between aggregate fidelity and extreme-event coverage. Multi-objective formulations provide a more principled alternative: they compute the full trade-off frontier and let the modeler choose, rather than committing to a fixed split.

5. **The choice of $\mathcal{A}$ often dominates the method's identity.** Methods are typically named after their search algorithm (k-means, MILP, genetic algorithm), but the other components — particularly $\mathcal{F}$ and $\mathcal{O}$ — often have a greater impact on the quality of the result. The framework redirects attention from *how* the search is conducted to *what* is being optimized and *how* quality is measured.

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
2. **Better proxies for $\mathcal{O}_\text{model}$.** Developing statistical objectives that are stronger predictors of model outcome fidelity — without requiring model-in-the-loop evaluation — remains a major research gap.
3. **Automated component selection.** Can the best $(\mathcal{F}, \mathcal{O}, \mathcal{S}, \mathcal{R}, \mathcal{A})$ tuple be selected automatically based on problem characteristics?
4. **$\mathcal{R}_\text{soft}$ adoption.** Blended representations promise higher fidelity but require ESM reformulation. Quantifying the fidelity gain is important for motivating this effort.

---

## 5. Conclusion

The field of representative period selection for energy time series has produced a rich and growing body of methods, each designed with care for specific use cases. However, these methods are typically presented as monolithic procedures, which obscures their shared structure and makes systematic comparison difficult.

The five-component framework proposed in this paper — Feature Space, Objective, Selection Space, Representation Model, and Search Algorithm — provides a common structure for understanding any TSA method. By decomposing methods into their fundamental choices, the framework:

- makes implicit assumptions explicit,
- enables direct, component-level comparison between methods,
- guides practitioners in assembling custom pipelines suited to their specific modeling questions, and
- provides an architectural blueprint for modular software design.

The framework does not claim that one component combination is universally superior. Rather, it provides the vocabulary and structure needed to make informed, transparent choices — moving the field from ad-hoc method selection toward systematic, principled design of representative period selection pipelines.

---

## References

*[To be populated with full citations from the literature. Key references include:]*

1. Hoffmann et al. (2020). "A Review on Time Series Aggregation Methods for Energy System Models." *Energies*, 13(3), 641.
2. Kotzur et al. (2022). "Time-series aggregation for the optimization of energy systems: Goals, challenges, approaches, and opportunities." *RSER*, 157.
3. Nahmmacher et al. (2016). "Selecting Representative Days for Capturing the Implications of Integrating Intermittent Renewables in Generation Expansion Planning."
4. Scott et al. (2022). "Representative period selection for power system planning using autoencoder-based dimensionality reduction." *arXiv:2204.13608*.
5. Sun et al. (2025). "An Interregional Optimization Approach for Time Series Aggregation in Continent-Scale Electricity System Models." NREL/TP-6A20-90183.
6. Teichgraeber & Brandt (2024). "On the Selection of Intermediate Length Representative Periods for Capacity Expansion." *arXiv:2401.02888*.
7. Hull Clustering with Blended Representative Periods (2025). *arXiv:2508.21641*.
8. Kotzur et al. (2018). "Impact of different time series aggregation methods on optimal energy system design."
9. Bahl et al. (2018). "Typical periods for two-stage synthesis by time-series aggregation with bounded error in objective function."
10. Poncelet et al. (2017). "Selecting Representative Days for Investment Planning Models."
