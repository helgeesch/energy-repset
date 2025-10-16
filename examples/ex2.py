import pandas as pd
from mesqual_repset.context import ProblemContext
from mesqual_repset.representation import KMedoidsClustersizeRepresentation
from mesqual_repset.time_slicer import TimeSlicer
from mesqual_repset.feature_engineering import StandardStatsFeatureEngineer
from mesqual_repset.objectives import ObjectiveSet
from mesqual_repset.problem import RepSetExperiment
from mesqual_repset.workflow import Workflow
from mesqual_repset.score_components import WassersteinFidelity, CorrelationFidelity
from mesqual_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from mesqual_repset.selection_policies import ParetoMaxMinStrategy
from mesqual_repset.combi_gens import GroupQuotaHierarchicalCombiGen

# Load raw time-series data
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)

# Define problem context with DAILY slicing (for high-resolution features)
child_slicer = TimeSlicer(unit="day")
context = ProblemContext(
    df_raw=df_raw,
    slicer=child_slicer,
    metadata={
        'experiment_name': 'ex2_hierarchical_seasonal',
        'notes': 'Hierarchical selection: day-level features, month-level selection with seasonal quotas'
    }
)

print(f"Problem Context created with {len(context.get_unique_slices())} daily slices.")

# Feature engineering: compute features per day
feature_engineer = StandardStatsFeatureEngineer()
context = feature_engineer.run(context)
print(f"Features computed for {len(context.df_features)} daily periods.")

# Define scoring and policy
objective_set = ObjectiveSet(
    weighted_score_components={
        'wasserstein': (0.5, WassersteinFidelity()),
        'correlation': (0.5, CorrelationFidelity()),
    }
)
policy = ParetoMaxMinStrategy()

# Hierarchical combination generator: select 4 MONTHS (1 per season), evaluate on DAYS
# Using the factory method with automatic seasonal grouping
combi_gen = GroupQuotaHierarchicalCombiGen.from_slicers_with_seasons(
    parent_k=4,  # Select 4 parent groups (months) total
    dt_index=df_raw.index,  # DatetimeIndex
    child_slicer=child_slicer,  # Daily slicing
    group_quota={'winter': 1, 'spring': 1, 'summer': 1, 'fall': 1}  # 1 month per season
)

days = context.get_unique_slices()
print(f"Hierarchical generator will evaluate {combi_gen.count(days)} combinations.")
print("Each combination = 4 months (1/season), evaluated on ~120 days total.\n")

# Search algorithm
search_algorithm = ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combi_gen)

# Representation model
representation_model = KMedoidsClustersizeRepresentation()

# Run workflow
workflow = Workflow(feature_engineer, search_algorithm, representation_model)
experiment = RepSetExperiment(context, workflow)
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
