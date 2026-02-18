"""Example 5: Multi-Objective Exploration.

This example demonstrates how different score components and selection policies
affect the outcome.  It sets up a rich ObjectiveSet with four components, then
runs the same search twice — once with ParetoMaxMinStrategy and once with
WeightedSumPolicy — to compare the winning selections and visualize trade-offs.

Diagnostics showcased (many not used in ex1/ex2):
  - ParetoScatter2D and ParetoScatterMatrix
  - ScoreContributionBars (side-by-side for both policies)
  - DistributionOverlayHistogram
  - FeatureDistributions
"""

import os
import pandas as pd

from energy_repset.context import ProblemContext
from energy_repset.time_slicer import TimeSlicer
from energy_repset.feature_engineering import (
    FeaturePipeline, StandardStatsFeatureEngineer, PCAFeatureEngineer,
)
from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import (
    WassersteinFidelity,
    CorrelationFidelity,
    DurationCurveFidelity,
    DiversityReward,
)
from energy_repset.selection_policies import ParetoMaxMinStrategy, WeightedSumPolicy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from energy_repset.representation import UniformRepresentationModel
from energy_repset.workflow import Workflow
from energy_repset.problem import RepSetExperiment
from energy_repset.diagnostics.results import (
    ParetoScatter2D,
    ParetoScatterMatrix,
    ScoreContributionBars,
    ResponsibilityBars,
)
from energy_repset.diagnostics.score_components import (
    DistributionOverlayHistogram,
    DiurnalProfileOverlay,
    CorrelationDifferenceHeatmap,
)
from energy_repset.diagnostics.feature_space import FeatureDistributions

OUTPUT_DIR = 'docs/gallery/ex5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load data ---
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)

# --- 2. Problem context and features ---
slicer = TimeSlicer(unit="month")
context = ProblemContext(df_raw=df_raw, slicer=slicer)

feature_pipeline = FeaturePipeline(engineers={
    'stats': StandardStatsFeatureEngineer(),
    'pca': PCAFeatureEngineer(),
})

# --- 3. Rich objective set with 4 components ---
# Each component captures a different aspect of representativeness:
#   wasserstein  — marginal distribution similarity  (direction: min)
#   correlation  — cross-variable correlation match   (direction: min)
#   duration_curve — duration-curve NRMSE             (direction: min)
#   diversity    — spread in feature space            (direction: max)
objective_set = ObjectiveSet({
    'wasserstein': (1.0, WassersteinFidelity()),
    'correlation': (1.0, CorrelationFidelity()),
    'duration_curve': (1.0, DurationCurveFidelity()),
    'diversity': (0.5, DiversityReward()),
})

k = 3
combi_gen = ExhaustiveCombiGen(k=k)
representation_model = UniformRepresentationModel()

# --- 4A. Run with ParetoMaxMinStrategy ---
print("=== Run A: ParetoMaxMinStrategy ===")
pareto_policy = ParetoMaxMinStrategy()
search_pareto = ObjectiveDrivenCombinatorialSearchAlgorithm(
    objective_set, pareto_policy, combi_gen
)
workflow_pareto = Workflow(feature_pipeline, search_pareto, representation_model)
experiment_pareto = RepSetExperiment(context, workflow_pareto)
result_pareto = experiment_pareto.run()
print(f"Selection: {result_pareto.selection}")
print(f"Scores:    {result_pareto.scores}")

# --- 4B. Run with WeightedSumPolicy ---
# Re-use the already-computed features from run A so we skip redundant work.
print("\n=== Run B: WeightedSumPolicy (robust_minmax normalization) ===")
weighted_policy = WeightedSumPolicy(normalization='robust_minmax')
search_weighted = ObjectiveDrivenCombinatorialSearchAlgorithm(
    objective_set, weighted_policy, combi_gen
)
workflow_weighted = Workflow(feature_pipeline, search_weighted, representation_model)
experiment_weighted = RepSetExperiment(
    experiment_pareto.feature_context, workflow_weighted
)
result_weighted = experiment_weighted.run()
print(f"Selection: {result_weighted.selection}")
print(f"Scores:    {result_weighted.scores}")

# --- 5. Compare ---
same = result_pareto.selection == result_weighted.selection
print(f"\nSame selection? {same}")

# --- 6. Diagnostics ---
print("\n--- Generating diagnostics ---")
feature_context = experiment_pareto.feature_context

# 6a. Pareto front visualizations (from the Pareto run)
fig_pareto_2d = ParetoScatter2D(
    objective_x='wasserstein', objective_y='correlation'
).plot(search_algorithm=search_pareto, selected_combination=result_pareto.selection)
fig_pareto_2d.update_layout(title='Ex5: Pareto Front — Wasserstein vs Correlation')
fig_pareto_2d.write_html(f'{OUTPUT_DIR}/pareto_scatter_2d.html')
print(f"Saved: {OUTPUT_DIR}/pareto_scatter_2d.html")

fig_pareto_matrix = ParetoScatterMatrix().plot(
    search_algorithm=search_pareto,
    selected_combination=result_pareto.selection,
)
fig_pareto_matrix.update_layout(title='Ex5: Pareto Scatter Matrix')
fig_pareto_matrix.write_html(f'{OUTPUT_DIR}/pareto_scatter_matrix.html')
print(f"Saved: {OUTPUT_DIR}/pareto_scatter_matrix.html")

# 6b. Score contribution bars for both policies
for label, result in [('pareto', result_pareto), ('weighted_sum', result_weighted)]:
    fig_scores = ScoreContributionBars().plot(result.scores, normalize=True)
    fig_scores.update_layout(title=f'Ex5: Score Contributions — {label}')
    fig_scores.write_html(f'{OUTPUT_DIR}/score_contributions_{label}.html')
    print(f"Saved: {OUTPUT_DIR}/score_contributions_{label}.html")

# 6c. Responsibility bars for both policies
for label, result in [('pareto', result_pareto), ('weighted_sum', result_weighted)]:
    fig_resp = ResponsibilityBars().plot(result.weights, show_uniform_reference=True)
    fig_resp.update_layout(title=f'Ex5: Weights — {label}')
    fig_resp.write_html(f'{OUTPUT_DIR}/responsibility_{label}.html')
    print(f"Saved: {OUTPUT_DIR}/responsibility_{label}.html")

# 6d. Distribution overlay histograms (Pareto selection)
selected_indices = slicer.get_indices_for_slice_combi(df_raw.index, result_pareto.selection)
df_selection = df_raw.loc[selected_indices]

for var in df_raw.columns:
    fig_hist = DistributionOverlayHistogram().plot(
        df_full=df_raw[var],
        df_selection=df_selection[var],
        nbins=40,
    )
    fig_hist.update_layout(title=f'Ex5: Distribution Overlay — {var}')
    fig_hist.write_html(f'{OUTPUT_DIR}/histogram_{var}.html')
print(f"Saved: {OUTPUT_DIR}/histogram_*.html (one per variable)")

# 6e. Feature distributions
fig_feat = FeatureDistributions().plot(feature_context.df_features, nbins=20, cols=4)
fig_feat.update_layout(title='Ex5: Feature Distributions')
fig_feat.write_html(f'{OUTPUT_DIR}/feature_distributions.html')
print(f"Saved: {OUTPUT_DIR}/feature_distributions.html")

# 6f. Diurnal profiles and correlation difference (Pareto selection)
fig_diurnal = DiurnalProfileOverlay().plot(
    df_raw, df_selection, variables=list(df_raw.columns)
)
fig_diurnal.update_layout(title='Ex5: Diurnal Profiles -- Full vs Selection')
fig_diurnal.write_html(f'{OUTPUT_DIR}/diurnal_profiles.html')
print(f"Saved: {OUTPUT_DIR}/diurnal_profiles.html")

fig_corr_diff = CorrelationDifferenceHeatmap().plot(
    df_raw, df_selection, method='pearson', show_lower_only=True
)
fig_corr_diff.update_layout(title='Ex5: Correlation Difference')
fig_corr_diff.write_html(f'{OUTPUT_DIR}/correlation_difference.html')
print(f"Saved: {OUTPUT_DIR}/correlation_difference.html")

print("\n--- All diagnostics complete ---")
