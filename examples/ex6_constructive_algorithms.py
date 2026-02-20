"""Example 6: Constructive Algorithms.

Demonstrates the three constructive search algorithms now available in
energy-repset.  Unlike the Generate-and-Test workflow (examples 1--5), these
algorithms build solutions directly using their own internal objectives.

The example runs:
  1. Hull Clustering   — greedy projection-error minimization (monthly)
  2. CTPC              — contiguity-constrained hierarchical clustering (monthly)
  3. Snippet Algorithm  — greedy p-median multi-day subsequences (daily, 7-day)

All three operate on the same underlying hourly dataset.
"""

import os
import pandas as pd
import energy_repset as rep
import energy_repset.diagnostics as diag

OUTPUT_DIR = 'docs/gallery/ex6'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load data ---
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)

print("=" * 60)
print("EXAMPLE 6: Constructive Algorithms")
print("=" * 60)

# =====================================================================
# 2. Hull Clustering (monthly slicing, k=3, convex hull)
# =====================================================================
print("\n--- Hull Clustering ---")
print("Selects months that span the convex hull of the feature space.")
print("Weights are computed externally via BlendedRepresentationModel.\n")

context_monthly = rep.ProblemContext(df_raw=df_raw, slicer=rep.TimeSlicer(unit="month"))

workflow_hull = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.HullClusteringSearch(k=3, hull_type='convex'),
    representation_model=rep.BlendedRepresentationModel(blend_type='convex'),
)
experiment_hull = rep.RepSetExperiment(context_monthly, workflow_hull)
result_hull = experiment_hull.run()

print(f"Selection:        {result_hull.selection}")
print(f"Projection error: {result_hull.scores['projection_error']:.4f}")
print(f"Weights type:     {type(result_hull.weights).__name__}")
if isinstance(result_hull.weights, pd.DataFrame):
    agg = result_hull.weights.sum(axis=0)
    agg_norm = (agg / agg.sum()).to_dict()
    print(f"Weights (agg):    { {str(k): round(v, 3) for k, v in agg_norm.items()} }")
else:
    print(f"Weights:          {result_hull.weights}")

# Visualizations: responsibility bars (aggregated blended weights)
if isinstance(result_hull.weights, pd.DataFrame):
    agg = result_hull.weights.sum(axis=0)
    weights_hull_agg = (agg / agg.sum()).to_dict()
else:
    weights_hull_agg = result_hull.weights

fig = diag.ResponsibilityBars().plot(weights_hull_agg, show_uniform_reference=True)
fig.update_layout(title='Hull Clustering: Responsibility Weights (Blended, Aggregated)')
fig.write_html(f'{OUTPUT_DIR}/hull_responsibility.html')

# Feature scatter with selection
feature_ctx_hull = experiment_hull.feature_context
cols = list(feature_ctx_hull.df_features.columns[:2])
fig = diag.FeatureSpaceScatter2D().plot(
    feature_ctx_hull.df_features, x=cols[0], y=cols[1], selection=result_hull.selection
)
fig.update_layout(title='Hull Clustering: Feature Space with Selection')
fig.write_html(f'{OUTPUT_DIR}/hull_feature_scatter.html')

# ECDF overlay for load
slicer_monthly = rep.TimeSlicer(unit="month")
selected_idx_hull = slicer_monthly.get_indices_for_slice_combi(df_raw.index, result_hull.selection)
df_sel_hull = df_raw.loc[selected_idx_hull]
fig = diag.DistributionOverlayECDF().plot(df_raw['load'], df_sel_hull['load'])
fig.update_layout(title='Hull Clustering: ECDF Overlay -- Load')
fig.write_html(f'{OUTPUT_DIR}/hull_ecdf_load.html')
print("Saved Hull Clustering plots.")

# =====================================================================
# 3. CTPC (monthly slicing, k=4, ward linkage)
# =====================================================================
print("\n--- CTPC (Chronological Time-Period Clustering) ---")
print("Merges only temporally adjacent periods into contiguous segments.")
print("Weights are pre-computed as segment size fractions.\n")

workflow_ctpc = rep.Workflow(
    feature_engineer=rep.StandardStatsFeatureEngineer(),
    search_algorithm=rep.CTPCSearch(k=4, linkage='ward'),
    representation_model=rep.UniformRepresentationModel(),  # placeholder, skipped
)
experiment_ctpc = rep.RepSetExperiment(context_monthly, workflow_ctpc)
result_ctpc = experiment_ctpc.run()

print(f"Selection:  {result_ctpc.selection}")
print(f"WCSS:       {result_ctpc.scores['wcss']:.4f}")
print(f"Weights:    { {str(k): round(v, 3) for k, v in result_ctpc.weights.items()} }")

# Segment info
if 'segments' in result_ctpc.diagnostics:
    print("\nContiguous segments:")
    for seg in result_ctpc.diagnostics['segments']:
        print(f"  {seg['start']} -- {seg['end']}  "
              f"(size={seg['size']}, representative={seg['representative']})")

# Visualizations: responsibility bars
fig = diag.ResponsibilityBars().plot(result_ctpc.weights, show_uniform_reference=True)
fig.update_layout(title='CTPC: Responsibility Weights (Segment Fractions)')
fig.write_html(f'{OUTPUT_DIR}/ctpc_responsibility.html')

# Feature scatter with selection
feature_ctx_ctpc = experiment_ctpc.feature_context
cols = list(feature_ctx_ctpc.df_features.columns[:2])
fig = diag.FeatureSpaceScatter2D().plot(
    feature_ctx_ctpc.df_features, x=cols[0], y=cols[1], selection=result_ctpc.selection
)
fig.update_layout(title='CTPC: Feature Space with Selection')
fig.write_html(f'{OUTPUT_DIR}/ctpc_feature_scatter.html')

# ECDF overlay for load
selected_idx_ctpc = slicer_monthly.get_indices_for_slice_combi(df_raw.index, result_ctpc.selection)
df_sel_ctpc = df_raw.loc[selected_idx_ctpc]
fig = diag.DistributionOverlayECDF().plot(df_raw['load'], df_sel_ctpc['load'])
fig.update_layout(title='CTPC: ECDF Overlay -- Load')
fig.write_html(f'{OUTPUT_DIR}/ctpc_ecdf_load.html')
print("Saved CTPC plots.")

# =====================================================================
# 4. Snippet Algorithm (daily slicing, k=4, 7-day periods)
# =====================================================================
print("\n--- Snippet Algorithm ---")
print("Selects multi-day subsequences via greedy p-median on daily profiles.")
print("Weights are pre-computed as assignment fractions.\n")

context_daily = rep.ProblemContext(df_raw=df_raw, slicer=rep.TimeSlicer(unit="day"))

workflow_snippet = rep.Workflow(
    feature_engineer=rep.DirectProfileFeatureEngineer(),
    search_algorithm=rep.SnippetSearch(k=4, period_length_days=7, step_days=7),
    representation_model=rep.UniformRepresentationModel(),  # placeholder, skipped
)
experiment_snippet = rep.RepSetExperiment(context_daily, workflow_snippet)
result_snippet = experiment_snippet.run()

print(f"Selection (start days): {result_snippet.selection}")
print(f"Total distance:         {result_snippet.scores['total_distance']:.4f}")
print(f"Weights:                { {str(k): round(v, 3) for k, v in result_snippet.weights.items()} }")

# Visualizations: responsibility bars
fig = diag.ResponsibilityBars().plot(result_snippet.weights, show_uniform_reference=True)
fig.update_layout(title='Snippet: Responsibility Weights (Assignment Fractions)')
fig.write_html(f'{OUTPUT_DIR}/snippet_responsibility.html')

# ECDF overlay for load
slicer_daily = rep.TimeSlicer(unit="day")
selected_idx_snippet = slicer_daily.get_indices_for_slice_combi(
    df_raw.index, result_snippet.selection
)
df_sel_snippet = df_raw.loc[selected_idx_snippet]
fig = diag.DistributionOverlayECDF().plot(df_raw['load'], df_sel_snippet['load'])
fig.update_layout(title='Snippet: ECDF Overlay -- Load')
fig.write_html(f'{OUTPUT_DIR}/snippet_ecdf_load.html')
print("Saved Snippet plots.")

# =====================================================================
# 5. Summary comparison
# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Algorithm':<25} {'k':>3} {'Internal Score':>20} {'Value':>10}")
print("-" * 60)
print(f"{'Hull Clustering':<25} {'3':>3} {'projection_error':>20} "
      f"{result_hull.scores['projection_error']:>10.4f}")
print(f"{'CTPC':<25} {'4':>3} {'wcss':>20} "
      f"{result_ctpc.scores['wcss']:>10.4f}")
print(f"{'Snippet':<25} {'4':>3} {'total_distance':>20} "
      f"{result_snippet.scores['total_distance']:>10.4f}")
print("\nAll plots saved to", OUTPUT_DIR)
