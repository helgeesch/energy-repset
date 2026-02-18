"""Example 3: Getting Started — Minimal end-to-end workflow.

This is the simplest possible energy-repset example. It selects 4
representative months from a year of hourly time-series data using a single
objective (Wasserstein fidelity), a weighted-sum policy, and uniform weights.

The five pillars of the framework appear in order:
  F — Feature engineering (StandardStatsFeatureEngineer)
  O — Objective (single WassersteinFidelity score component)
  S — Selection space (ExhaustiveCombiGen over monthly slices)
  R — Representation model (UniformRepresentationModel: equal 1/k weights)
  A — Search algorithm (ObjectiveDrivenCombinatorialSearchAlgorithm)
"""

import os
import pandas as pd

from energy_repset.context import ProblemContext
from energy_repset.time_slicer import TimeSlicer
from energy_repset.feature_engineering import StandardStatsFeatureEngineer
from energy_repset.objectives import ObjectiveSet
from energy_repset.score_components import WassersteinFidelity
from energy_repset.selection_policies import WeightedSumPolicy
from energy_repset.combi_gens import ExhaustiveCombiGen
from energy_repset.search_algorithms import ObjectiveDrivenCombinatorialSearchAlgorithm
from energy_repset.representation import UniformRepresentationModel
from energy_repset.workflow import Workflow
from energy_repset.problem import RepSetExperiment
from energy_repset.diagnostics.results import ResponsibilityBars

OUTPUT_DIR = 'docs/gallery/ex3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load data ---
url = "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
df_raw = pd.read_csv(url, index_col=0, parse_dates=True).rename_axis('variable', axis=1)
df_raw = df_raw.drop('prices', axis=1)

# --- 2. Define the problem context ---
# The TimeSlicer divides the year into monthly candidate periods.
slicer = TimeSlicer(unit="month")
context = ProblemContext(df_raw=df_raw, slicer=slicer)
print(f"Candidate slices: {context.get_unique_slices()}")

# --- 3. Pillar F: Feature engineering ---
# Compute statistical summaries (mean, std, min, max, ...) per month and variable.
feature_engineer = StandardStatsFeatureEngineer()

# --- 4. Pillar O: Objective ---
# A single score component: how well does the selection's marginal distribution
# match the full year?  Lower Wasserstein distance = better.
objective_set = ObjectiveSet({
    'wasserstein': (1.0, WassersteinFidelity()),
})

# --- 5. Pillar S + A: Search ---
# Exhaustively evaluate every 4-of-12 monthly combination (495 candidates).
k = 4
combi_gen = ExhaustiveCombiGen(k=k)
policy = WeightedSumPolicy()
search_algorithm = ObjectiveDrivenCombinatorialSearchAlgorithm(
    objective_set, policy, combi_gen
)

# --- 6. Pillar R: Representation model ---
# Uniform weights: each selected month represents 1/k of the year.
representation_model = UniformRepresentationModel()

# --- 7. Run the workflow ---
workflow = Workflow(feature_engineer, search_algorithm, representation_model)
experiment = RepSetExperiment(context, workflow)
result = experiment.run()

# --- 8. Inspect results ---
print(f"\nSelected months: {result.selection}")
print(f"Weights: {result.weights}")
print(f"Wasserstein score: {result.scores['wasserstein']:.4f}")

# --- 9. Diagnostic: responsibility bar chart ---
fig = ResponsibilityBars().plot(result.weights, show_uniform_reference=True)
fig.update_layout(title='Ex3: Responsibility Weights (Uniform)')
fig.write_html(f'{OUTPUT_DIR}/responsibility_weights.html')
print(f"\nPlot saved to {OUTPUT_DIR}/responsibility_weights.html")
