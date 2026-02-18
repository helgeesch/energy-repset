import os
import pandas as pd
from energy_repset.context import ProblemContext
from energy_repset.representation import KMedoidsClustersizeRepresentation
from energy_repset.time_slicer import TimeSlicer
from energy_repset.feature_engineering import FeaturePipeline, StandardStatsFeatureEngineer, PCAFeatureEngineer
from energy_repset.objectives import ObjectiveSet
from energy_repset.problem import RepSetExperiment
from energy_repset.workflow import Workflow
from energy_repset.score_components import (
    WassersteinFidelity,
    CorrelationFidelity,
    DiversityReward,
    CentroidBalance
)
from energy_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from energy_repset.selection_policies import ParetoMaxMinStrategy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.diagnostics.feature_space import (
    FeatureSpaceScatter2D,
    FeatureSpaceScatter3D,
    FeatureSpaceScatterMatrix,
    PCAVarianceExplained,
    FeatureCorrelationHeatmap,
    FeatureDistributions,
)
from energy_repset.diagnostics.score_components import (
    DistributionOverlayECDF,
    CorrelationDifferenceHeatmap,
    DiurnalProfileOverlay,
)
from energy_repset.diagnostics.results import ResponsibilityBars

OUTPUT_DIR = 'docs/gallery/ex1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Initial Data Loading ---
# Load the raw time-series data.
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)


# --- 2. Defining the Problem Context ---
# The ProblemContext is the central hub, holding the raw data and the
# definition of our candidate slices.
slicer = TimeSlicer(unit="month")
context = ProblemContext(
    df_raw=df_raw,
    slicer=slicer,
    metadata={
        'experiment_name': 'ex1_monthly_selection',
        'notes': 'Basic example with monthly slicing and uniform weights'
    }
)

print(f"Problem Context created with {len(context.get_unique_slices())} candidate slices.")


# --- 3. Pillar 1: Feature Engineering ---
# We define a pipeline to transform the raw data into a meaningful feature space.
# Here, we chain a statistical feature engineer with a PCA dimensionality reducer.
feature_pipeline = FeaturePipeline(
    engineers={
        'stats': StandardStatsFeatureEngineer(),
        'pca': PCAFeatureEngineer(),
    }
)

# The pipeline is run on the context, which updates it with the new features.
context = feature_pipeline.run(context)

print("\n--- Feature Space Diagnostics ---")

# Visualize PCA variance explained
pca_variance_plot = PCAVarianceExplained(feature_pipeline.engineers['pca'])
fig_pca_variance = pca_variance_plot.plot(show_cumulative=True)
fig_pca_variance.update_layout(title='PCA Variance Explained')
fig_pca_variance.write_html(f'{OUTPUT_DIR}/output_pca_variance.html')
print("Generated: output_pca_variance.html")

# 2D scatter plot of PCA space
scatter_2d = FeatureSpaceScatter2D()
fig_scatter_2d = scatter_2d.plot(
    df_features=context.df_features,
    x='pc_0',
    y='pc_1',
)
fig_scatter_2d.update_layout(title='Feature Space: PC0 vs PC1')
fig_scatter_2d.write_html(f'{OUTPUT_DIR}/output_feature_scatter_2d.html')
print("Generated: output_feature_scatter_2d.html")

# 3D scatter plot of PCA space
scatter_3d = FeatureSpaceScatter3D()
fig_scatter_3d = scatter_3d.plot(
    df_features=context.df_features,
    x='pc_0',
    y='pc_1',
    z='pc_2',
)
fig_scatter_3d.update_layout(title='Feature Space: 3D PCA Projection')
fig_scatter_3d.write_html(f'{OUTPUT_DIR}/output_feature_scatter_3d.html')
print("Generated: output_feature_scatter_3d.html")

# Feature correlation heatmap
corr_heatmap = FeatureCorrelationHeatmap()
fig_corr = corr_heatmap.plot(
    df_features=context.df_features,
    method='pearson',
)
fig_corr.update_layout(title='Feature Correlation Matrix')
fig_corr.write_html(f'{OUTPUT_DIR}/output_feature_correlation.html')
print("Generated: output_feature_correlation.html")

# Scatter matrix for first 4 PCA components
scatter_matrix = FeatureSpaceScatterMatrix()
fig_splom = scatter_matrix.plot(
    df_features=context.df_features,
    dimensions=['pc_0', 'pc_1', 'pc_2', 'pc_3'],
)
fig_splom.update_layout(title='PCA Feature Space Scatter Matrix')
fig_splom.write_html(f'{OUTPUT_DIR}/output_feature_scatter_matrix.html')
print("Generated: output_feature_scatter_matrix.html")

# Feature distributions
fig_feat = FeatureDistributions().plot(context.df_features, nbins=20, cols=4)
fig_feat.update_layout(title='Feature Distributions')
fig_feat.write_html(f'{OUTPUT_DIR}/output_feature_distributions.html')
print("Generated: output_feature_distributions.html")

# --- 4. Pillars 2 & 3: ObjectiveSet and Selection Policy ---
# Define the scoring rubric (ObjectiveSet) and the rule for picking a winner (Policy).

objective_set = ObjectiveSet(
    weighted_score_components={
        'wasserstein': (0.5, WassersteinFidelity()),
        'correlation': (0.5, CorrelationFidelity()),
        # 'diversity': (0.5, DiversityReward()),
        'centroid_balance': (0.5, CentroidBalance()),
    }
)
policy = ParetoMaxMinStrategy()

# --- 5. Pillar 5: Search Algorithm ---
# Define the engine that will search for the best subset. Here, we use a
# combinatorial search that is constrained to pick at least one week per season.
k = 3
combi_gen = ExhaustiveCombiGen(k=k)
search_algorithm = ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combi_gen)

# --- 6. Pillar 4: Representation Model ---
# Define how the final selected weeks will represent the full year.
representation_model = KMedoidsClustersizeRepresentation()

# --- 7. The Workflow: Executing the Workflow ---
workflow = Workflow(feature_pipeline, search_algorithm, representation_model)

experiment = RepSetExperiment(context, workflow)
result = experiment.run()

print("\n--- Workflow Complete ---")

# --- 8. Results and Diagnostics ---
# The final result is a standardized object containing all relevant information.
print(f"Selected Slices: {result.selection}")
print(f"Final Weights: {result.weights}")
print(f"Scores: {result.scores}")

print("\n--- Result Diagnostics ---")

# Visualize responsibility weights
responsibility_bars = ResponsibilityBars()
fig_responsibility = responsibility_bars.plot(
    weights=result.weights,
    show_uniform_reference=True,
)
fig_responsibility.update_layout(title='Responsibility Weights per Selected Month')
fig_responsibility.write_html(f'{OUTPUT_DIR}/output_responsibility_weights.html')
print("Generated: output_responsibility_weights.html")

print("\n--- Score Component Diagnostics ---")

# Get selected data for comparison
selected_indices = context.slicer.get_indices_for_slice_combi(context.df_raw.index, result.selection)
df_full = context.df_raw
df_selection = context.df_raw.loc[selected_indices]

# Distribution comparison for 'load' variable
ecdf_plot = DistributionOverlayECDF()
fig_ecdf = ecdf_plot.plot(
    df_full=df_full['load'],
    df_selection=df_selection['load'],
)
fig_ecdf.update_layout(title='Distribution Fidelity: Demand (ECDF)')
fig_ecdf.write_html(f'{OUTPUT_DIR}/output_distribution_ecdf_load.html')
print("Generated: output_distribution_ecdf_load.html")

# Correlation difference heatmap
corr_diff = CorrelationDifferenceHeatmap()
fig_corr_diff = corr_diff.plot(
    df_full=df_full,
    df_selection=df_selection,
    method='pearson',
    show_lower_only=True,
)
fig_corr_diff.update_layout(title='Correlation Difference: Selection - Full')
fig_corr_diff.write_html(f'{OUTPUT_DIR}/output_correlation_difference.html')
print("Generated: output_correlation_difference.html")

# Diurnal profile comparison
diurnal_plot = DiurnalProfileOverlay()
fig_diurnal = diurnal_plot.plot(
    df_full=df_full,
    df_selection=df_selection,
    variables=['load', 'onwind', 'offwind', 'solar'],
)
fig_diurnal.update_layout(title='Diurnal Profiles: Full vs Selection')
fig_diurnal.write_html(f'{OUTPUT_DIR}/output_diurnal_profiles.html')
print("Generated: output_diurnal_profiles.html")

# Visualize selection in feature space
scatter_2d_with_selection = FeatureSpaceScatter2D()
fig_scatter_selection = scatter_2d_with_selection.plot(
    df_features=context.df_features,
    x='pc_0',
    y='pc_1',
    selection=result.selection,
)
fig_scatter_selection.update_layout(title='Feature Space with Selected Months')
fig_scatter_selection.write_html(f'{OUTPUT_DIR}/output_feature_scatter_with_selection.html')
print("Generated: output_feature_scatter_with_selection.html")

print("\n--- All Diagnostics Complete ---")
print("Open the generated HTML files in a browser to view the interactive plots.")