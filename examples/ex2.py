import os
import pandas as pd
import energy_repset as rep
import energy_repset.diagnostics as diag

OUTPUT_DIR = 'docs/gallery/ex2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load raw time-series data
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)

# Define problem context with DAILY slicing (for high-resolution features)
child_slicer = rep.TimeSlicer(unit="day")
context = rep.ProblemContext(
    df_raw=df_raw,
    slicer=child_slicer,
    metadata={
        'experiment_name': 'ex2_hierarchical_seasonal',
        'notes': 'Hierarchical selection: day-level features, month-level selection with seasonal quotas'
    }
)

print(f"Problem Context created with {len(context.get_unique_slices())} daily slices.")

# Feature engineering: compute features per day
feature_engineer = rep.StandardStatsFeatureEngineer()
context = feature_engineer.run(context)
print(f"Features computed for {len(context.df_features)} daily periods.")

# Define scoring and policy
objective_set = rep.ObjectiveSet(
    weighted_score_components={
        'wasserstein': (0.5, rep.WassersteinFidelity()),
        'correlation': (0.5, rep.CorrelationFidelity()),
    }
)
policy = rep.ParetoMaxMinStrategy()

# Hierarchical combination generator: select 4 MONTHS (1 per season), evaluate on DAYS
# Using the factory method with automatic seasonal grouping
combi_gen = rep.GroupQuotaHierarchicalCombiGen.from_slicers_with_seasons(
    parent_k=4,  # Select 4 parent groups (months) total
    dt_index=df_raw.index,  # DatetimeIndex
    child_slicer=child_slicer,  # Daily slicing
    group_quota={'winter': 1, 'spring': 1, 'summer': 1, 'fall': 1}  # 1 month per season
)

days = context.get_unique_slices()
print(f"Hierarchical generator will evaluate {combi_gen.count(days)} combinations.")
print("Each combination = 4 months (1/season), evaluated on ~120 days total.\n")

# Search algorithm
search_algorithm = rep.ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combi_gen)

# Representation model
representation_model = rep.KMedoidsClustersizeRepresentation()

# Run workflow
workflow = rep.Workflow(feature_engineer, search_algorithm, representation_model)
experiment = rep.RepSetExperiment(context, workflow)
result = experiment.run()

print("\n--- Workflow Complete ---")
print(f"Selected Months (via daily slices): {result.selection[:5]}... (showing first 5 days)")
print(f"Total days in selection: {len(result.selection)}")
print(f"Final Weights: {result.weights}")
print(f"Scores: {result.scores}")

# Identify which months were selected
selected_months = set()
for day in result.selection:
    month = day.asfreq('M')
    selected_months.add(month)

print(f"\nSelected Months: {sorted(selected_months)}")

# Visualize responsibility weights
responsibility_bars = diag.ResponsibilityBars()
fig_responsibility = responsibility_bars.plot(
    weights=result.weights,
    show_uniform_reference=True,
)
fig_responsibility.update_layout(title='Responsibility Weights per Selected Month')
fig_responsibility.write_html(f'{OUTPUT_DIR}/output_responsibility_weights.html')
print("Generated: output_responsibility_weights.html")

# Visualize Pareto front (2D scatter)
pareto_scatter = diag.ParetoScatter2D(
    objective_x='wasserstein',
    objective_y='correlation'
)
fig_pareto = pareto_scatter.plot(
    search_algorithm=search_algorithm,
    selected_combination=result.selection
)
fig_pareto.update_layout(title='Pareto Front: Wasserstein vs Correlation')
fig_pareto.write_html(f'{OUTPUT_DIR}/output_pareto_scatter.html')
print("Generated: output_pareto_scatter.html")

# Visualize Pareto front (parallel coordinates)
pareto_parallel = diag.ParetoParallelCoordinates()
fig_parallel = pareto_parallel.plot(search_algorithm=search_algorithm)
fig_parallel.update_layout(title='Pareto Front: Parallel Coordinates')
fig_parallel.write_html(f'{OUTPUT_DIR}/output_pareto_parallel.html')
print("Generated: output_pareto_parallel.html")

# Visualize score contributions
score_bars = diag.ScoreContributionBars()
fig_scores = score_bars.plot(result.scores, normalize=True)
fig_scores.update_layout(title='Score Component Contributions (Normalized)')
fig_scores.write_html(f'{OUTPUT_DIR}/output_score_contributions.html')
print("Generated: output_score_contributions.html")

# Feature space scatter (first two feature columns)
cols = list(context.df_features.columns[:2])
fig_scatter = diag.FeatureSpaceScatter2D().plot(
    context.df_features, x=cols[0], y=cols[1], selection=result.selection
)
fig_scatter.update_layout(title='Feature Space with Selection')
fig_scatter.write_html(f'{OUTPUT_DIR}/output_feature_scatter.html')
print("Generated: output_feature_scatter.html")

# ECDF overlays per variable
selected_indices = child_slicer.get_indices_for_slice_combi(df_raw.index, result.selection)
df_selection = df_raw.loc[selected_indices]
for var in df_raw.columns:
    fig_ecdf = diag.DistributionOverlayECDF().plot(df_raw[var], df_selection[var])
    fig_ecdf.update_layout(title=f'ECDF Overlay -- {var}')
    fig_ecdf.write_html(f'{OUTPUT_DIR}/output_ecdf_{var}.html')
print("Generated: output_ecdf_*.html (one per variable)")
