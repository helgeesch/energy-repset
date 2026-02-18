import os
import pandas as pd
import energy_repset as rep
import energy_repset.diagnostics as diag

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
slicer = rep.TimeSlicer(unit="month")
context = rep.ProblemContext(
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
feature_pipeline = rep.FeaturePipeline(
    engineers={
        'stats': rep.StandardStatsFeatureEngineer(),
        'pca': rep.PCAFeatureEngineer(),
    }
)

# The pipeline is run on the context, which updates it with the new features.
context = feature_pipeline.run(context)

print("\n--- Feature Space Diagnostics ---")

# Visualize PCA variance explained
pca_variance_plot = diag.PCAVarianceExplained(feature_pipeline.engineers['pca'])
fig_pca_variance = pca_variance_plot.plot(show_cumulative=True)
fig_pca_variance.update_layout(title='PCA Variance Explained')
fig_pca_variance.write_html(f'{OUTPUT_DIR}/output_pca_variance.html')
print("Generated: output_pca_variance.html")

# 2D scatter plot of PCA space
scatter_2d = diag.FeatureSpaceScatter2D()
fig_scatter_2d = scatter_2d.plot(
    df_features=context.df_features,
    x='pc_0',
    y='pc_1',
)
fig_scatter_2d.update_layout(title='Feature Space: PC0 vs PC1')
fig_scatter_2d.write_html(f'{OUTPUT_DIR}/output_feature_scatter_2d.html')
print("Generated: output_feature_scatter_2d.html")

# 3D scatter plot of PCA space
scatter_3d = diag.FeatureSpaceScatter3D()
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
corr_heatmap = diag.FeatureCorrelationHeatmap()
fig_corr = corr_heatmap.plot(
    df_features=context.df_features,
    method='pearson',
)
fig_corr.update_layout(title='Feature Correlation Matrix')
fig_corr.write_html(f'{OUTPUT_DIR}/output_feature_correlation.html')
print("Generated: output_feature_correlation.html")

# Scatter matrix for first 4 PCA components
scatter_matrix = diag.FeatureSpaceScatterMatrix()
fig_splom = scatter_matrix.plot(
    df_features=context.df_features,
    dimensions=['pc_0', 'pc_1', 'pc_2', 'pc_3'],
)
fig_splom.update_layout(title='PCA Feature Space Scatter Matrix')
fig_splom.write_html(f'{OUTPUT_DIR}/output_feature_scatter_matrix.html')
print("Generated: output_feature_scatter_matrix.html")

# Feature distributions
fig_feat = diag.FeatureDistributions().plot(context.df_features, nbins=20, cols=4)
fig_feat.update_layout(title='Feature Distributions')
fig_feat.write_html(f'{OUTPUT_DIR}/output_feature_distributions.html')
print("Generated: output_feature_distributions.html")

# --- 4. Pillars 2 & 3: ObjectiveSet and Selection Policy ---
# Define the scoring rubric (ObjectiveSet) and the rule for picking a winner (Policy).

objective_set = rep.ObjectiveSet(
    weighted_score_components={
        'wasserstein': (0.5, rep.WassersteinFidelity()),
        'correlation': (0.5, rep.CorrelationFidelity()),
        # 'diversity': (0.5, rep.DiversityReward()),
        'centroid_balance': (0.5, rep.CentroidBalance()),
    }
)
policy = rep.ParetoMaxMinStrategy()

# --- 5. Pillar 5: Search Algorithm ---
# Define the engine that will search for the best subset. Here, we use a
# combinatorial search that is constrained to pick at least one week per season.
k = 3
combi_gen = rep.ExhaustiveCombiGen(k=k)
search_algorithm = rep.ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combi_gen)

# --- 6. Pillar 4: Representation Model ---
# Define how the final selected weeks will represent the full year.
representation_model = rep.KMedoidsClustersizeRepresentation()

# --- 7. The Workflow: Executing the Workflow ---
workflow = rep.Workflow(feature_pipeline, search_algorithm, representation_model)

experiment = rep.RepSetExperiment(context, workflow)
result = experiment.run()

print("\n--- Workflow Complete ---")

# --- 8. Results and Diagnostics ---
# The final result is a standardized object containing all relevant information.
print(f"Selected Slices: {result.selection}")
print(f"Final Weights: {result.weights}")
print(f"Scores: {result.scores}")

print("\n--- Result Diagnostics ---")

# Visualize responsibility weights
responsibility_bars = diag.ResponsibilityBars()
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
ecdf_plot = diag.DistributionOverlayECDF()
fig_ecdf = ecdf_plot.plot(
    df_full=df_full['load'],
    df_selection=df_selection['load'],
)
fig_ecdf.update_layout(title='Distribution Fidelity: Demand (ECDF)')
fig_ecdf.write_html(f'{OUTPUT_DIR}/output_distribution_ecdf_load.html')
print("Generated: output_distribution_ecdf_load.html")

# Correlation difference heatmap
corr_diff = diag.CorrelationDifferenceHeatmap()
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
diurnal_plot = diag.DiurnalProfileOverlay()
fig_diurnal = diurnal_plot.plot(
    df_full=df_full,
    df_selection=df_selection,
    variables=['load', 'onwind', 'offwind', 'solar'],
)
fig_diurnal.update_layout(title='Diurnal Profiles: Full vs Selection')
fig_diurnal.write_html(f'{OUTPUT_DIR}/output_diurnal_profiles.html')
print("Generated: output_diurnal_profiles.html")

# Visualize selection in feature space
scatter_2d_with_selection = diag.FeatureSpaceScatter2D()
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
