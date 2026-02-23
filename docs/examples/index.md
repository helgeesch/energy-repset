# Examples

Interactive Jupyter notebooks demonstrating the `energy-repset` framework, from basic usage to advanced multi-objective and constructive algorithm workflows.

---

## [Example 1: Getting Started](ex1_getting_started.ipynb)

The simplest end-to-end workflow. Selects 4 representative months using a single objective (Wasserstein fidelity) and uniform weights.

---

## [Example 2: Feature Space Exploration](ex2_feature_space.ipynb)

Chains statistical summaries with PCA dimensionality reduction, then uses multi-objective selection (Wasserstein + correlation + centroid balance) with KMedoids cluster-size weights.

---

## [Example 3: Hierarchical Seasonal Selection](ex3_hierarchical_selection.ipynb)

Daily-resolution features with monthly-level selection under seasonal constraints. Uses `GroupQuotaHierarchicalCombiGen` to enforce one month per season.

---

## [Example 4: Comparing Representation Models](ex4_representation_models.ipynb)

Same selection, three different representation models: Uniform, KMedoids cluster-size, and Blended (soft assignment). Compares how each distributes responsibility weights.

---

## [Example 5: Multi-Objective Exploration](ex5_multi_objective.ipynb)

Four-component objective with `ParetoMaxMinStrategy` vs `WeightedSumPolicy`. Includes Pareto front visualization and score contribution analysis.

---

## [Example 6: Constructive Algorithms](ex6_constructive_algorithms.ipynb)

Three constructive algorithms -- Hull Clustering, CTPC, and Snippet -- that build solutions using their own internal objectives, bypassing the Generate-and-Test workflow.

---

## [Example 8: Genetic Algorithm Search](ex8_genetic_algorithm.ipynb)

Scalable search with genetic algorithms and random sampling. Compares exhaustive, random, and GA-based selection using weighted-sum and NSGA-II fitness strategies.

---

## Running the Notebooks

```bash
# Install the package in editable mode
pip install -e .

# Launch Jupyter
jupyter notebook docs/examples/
```
