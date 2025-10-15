# =============================================================================
# A Dummy Workflow for the RepSet v2 Package
#
# This script illustrates the core workflow and interaction of the abstract
# modules designed for the repset-v2 package. It demonstrates how a user
# would define a problem, execute a search, and analyze the results.
# =============================================================================

import pandas as pd
from mesqual_repset.context import ProblemContext
from mesqual_repset.representation import UniformRepresentationModel
from mesqual_repset.time_slicer import TimeSlicer
from mesqual_repset.feature_engineering import FeaturePipeline, StandardStatsFeatureEngineer
from mesqual_repset.objectives import ObjectiveSet
from mesqual_repset.problem import RepSetExperiment
from mesqual_repset.workflow import Workflow
from mesqual_repset.score_components import WassersteinFidelity, CorrelationFidelity
from mesqual_repset.score_components.todo import DiversityReward, CentroidBalance
from mesqual_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from mesqual_repset.selection_policies import ParetoMaxMinStrategy
from mesqual_repset.combination_generator import ExhaustiveCombinationGenerator
# from mesqual_repset.diagnostics import Visualizer

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
    variable_weights={c: 1 for c in df_raw.columns},  # TODO: respect variable weights in algorithm modules
    feature_weights={}  # TODO: respect feature weights
)

print(f"Problem Context created with {len(context.get_unique_slices())} candidate slices.")


# --- 3. Pillar 1: Feature Engineering ---
# We define a pipeline to transform the raw data into a meaningful feature space.
# Here, we chain a statistical feature engineer with a PCA dimensionality reducer.
feature_pipeline = FeaturePipeline(
    engineers=[StandardStatsFeatureEngineer()]
)

# The pipeline is run on the context, which updates it with the new features.
context = feature_pipeline.run(context)

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
combo_gen = ExhaustiveCombinationGenerator(k=k)  # TODO: assess whether to bind k here or not?
search_algorithm = ObjectiveDrivenCombinatorialSearchAlgorithm(objective_set, policy, combo_gen)

# --- 6. Pillar 4: Representation Model ---
# Define how the final selected weeks will represent the full year.
representation_model = UniformRepresentationModel()

# --- 7. The Workflow: Executing the Workflow ---
workflow = Workflow(feature_pipeline, search_algorithm, representation_model, k=k)

experiment = RepSetExperiment(context, workflow)
result = experiment.run()

print("\n--- Workflow Complete ---")

# --- 8. Results and Diagnostics ---
# The final result is a standardized object containing all relevant information.
print(f"Selected Slices: {result.selection}")
print(f"Final Weights: {result.weights}")
print(f"Scores: {result.scores}")

# Use a diagnostic tool to visualize the outcome.
# visualizer = Visualizer(context)
# fig = visualizer.plot_selection_in_pca(result)
# # fig.show() # In a real script, this would display an interactive plot.
# print("\nDiagnostic plot generated.")