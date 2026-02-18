"""Example 4: Comparing Representation Models.

This example runs a single search to find the best 3-month selection, then
applies three different representation models to the same winning selection:

  1. UniformRepresentationModel   — equal 1/k weights
  2. KMedoidsClustersizeRepresentation — cluster-proportional weights
  3. BlendedRepresentationModel   — soft assignment (weight matrix)

The output compares how each model distributes responsibility across the
selected months, illustrating the R (representation) pillar of the framework.
"""

import os
import pandas as pd
import plotly.express as px
import energy_repset as rep
import energy_repset.diagnostics as diag

OUTPUT_DIR = 'docs/gallery/ex4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load data ---
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)

# --- 2. Problem context and feature engineering ---
slicer = rep.TimeSlicer(unit="month")
context = rep.ProblemContext(df_raw=df_raw, slicer=slicer)

feature_pipeline = rep.FeaturePipeline(engineers={
    'stats': rep.StandardStatsFeatureEngineer(),
    'pca': rep.PCAFeatureEngineer(),
})

# --- 3. Objective set and search ---
k = 3
objective_set = rep.ObjectiveSet({
    'wasserstein': (1.0, rep.WassersteinFidelity()),
    'correlation': (1.0, rep.CorrelationFidelity()),
})
policy = rep.WeightedSumPolicy(normalization='robust_minmax')
combi_gen = rep.ExhaustiveCombiGen(k=k)
search_algorithm = rep.ObjectiveDrivenCombinatorialSearchAlgorithm(
    objective_set, policy, combi_gen
)

# Run with uniform weights first (we only need the search result).
workflow = rep.Workflow(feature_pipeline, search_algorithm, rep.UniformRepresentationModel())
experiment = rep.RepSetExperiment(context, workflow)
result = experiment.run()

selection = result.selection
print(f"Selected months: {selection}")
print(f"Scores: {result.scores}")

# --- 4. Apply three representation models to the same selection ---
feature_context = experiment.feature_context

# Model A: Uniform
uniform_model = rep.UniformRepresentationModel()
uniform_model.fit(feature_context)
weights_uniform = uniform_model.weigh(selection)

# Model B: KMedoids cluster-size
kmedoids_model = rep.KMedoidsClustersizeRepresentation()
kmedoids_model.fit(feature_context)
weights_kmedoids = kmedoids_model.weigh(selection)

# Model C: Blended (soft assignment)
blended_model = rep.BlendedRepresentationModel(blend_type='convex')
blended_model.fit(feature_context)
weights_blended_df = blended_model.weigh(selection)

# --- 5. Print weight comparison ---
print("\n--- Weight Comparison ---")
print(f"{'Month':<12} {'Uniform':>10} {'KMedoids':>10}")
print("-" * 34)
for s in selection:
    print(f"{str(s):<12} {weights_uniform[s]:>10.3f} {weights_kmedoids[s]:>10.3f}")

print(f"\nBlended weight matrix ({weights_blended_df.shape[0]} rows x "
      f"{weights_blended_df.shape[1]} cols):")
print(weights_blended_df.round(3).to_string())

# Aggregate blended weights: sum each column to get the total responsibility
# per representative, then normalize so weights sum to 1.0.
blended_column_sums = weights_blended_df.sum(axis=0)
weights_blended_agg = (blended_column_sums / blended_column_sums.sum()).to_dict()
print(f"\nBlended (aggregated, normalized): {weights_blended_agg}")

# --- 6. Diagnostics: side-by-side responsibility bars ---
models = {
    'Uniform': weights_uniform,
    'KMedoids': weights_kmedoids,
    'Blended (aggregated)': weights_blended_agg,
}

for label, weights in models.items():
    fig = diag.ResponsibilityBars().plot(weights, show_uniform_reference=True)
    fig.update_layout(title=f'Ex4: Responsibility Weights — {label}')
    filename = f"responsibility_{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}.html"
    fig.write_html(f'{OUTPUT_DIR}/{filename}')
    print(f"Saved: {OUTPUT_DIR}/{filename}")

# --- 7. Feature scatter with selection ---
fig_scatter = diag.FeatureSpaceScatter2D().plot(
    feature_context.df_features, x='pc_0', y='pc_1', selection=selection
)
fig_scatter.update_layout(title='Ex4: Feature Space with Selection')
fig_scatter.write_html(f'{OUTPUT_DIR}/feature_scatter_selection.html')
print(f"Saved: {OUTPUT_DIR}/feature_scatter_selection.html")

# --- 8. ECDF overlays per variable ---
selected_indices = slicer.get_indices_for_slice_combi(df_raw.index, selection)
df_selection = df_raw.loc[selected_indices]
for var in df_raw.columns:
    fig_ecdf = diag.DistributionOverlayECDF().plot(df_raw[var], df_selection[var])
    fig_ecdf.update_layout(title=f'Ex4: ECDF Overlay -- {var}')
    fig_ecdf.write_html(f'{OUTPUT_DIR}/ecdf_{var}.html')
print(f"Saved: {OUTPUT_DIR}/ecdf_*.html (one per variable)")

# --- 9. Blended weight heatmap ---
heatmap_df = weights_blended_df.copy()
heatmap_df.index = heatmap_df.index.astype(str)
heatmap_df.columns = heatmap_df.columns.astype(str)

fig_heatmap = px.imshow(
    heatmap_df.T,
    labels=dict(x='Original Month', y='Representative', color='Weight'),
    color_continuous_scale='Blues',
    aspect='auto',
    title='Ex4: Blended Weight Matrix (rows = representatives, cols = all months)',
)
fig_heatmap.write_html(f'{OUTPUT_DIR}/blended_heatmap.html')
print(f"Saved: {OUTPUT_DIR}/blended_heatmap.html")
