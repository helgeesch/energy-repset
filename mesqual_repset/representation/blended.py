from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mesqual_repset.context import ProblemContext
from mesqual_repset.representation.representation import RepresentationModel
from mesqual_repset.types import SliceCombination


class BlendedRepresentationModel(RepresentationModel):
    """
    Assigns weights using a blended representation (R_soft).

    Each original slice in the full dataset is represented as a unique
    weighted combination of all the selected representatives. This is found by
    solving a small optimization problem for each original slice.

    The output is a DataFrame where rows are the original slice labels,
    columns are the selected representative labels, and values are the weights.
    """

    def __init__(self, blend_type: str = 'convex'):
        """
        Parameters
        ----------
        blend_type : str, optional
            The type of blend to perform. 'convex' is the most common,
            ensuring weights are non-negative and sum to 1.
            (default is 'convex')
        """
        if blend_type != 'convex':
            raise NotImplementedError("Only 'convex' blend type is currently supported.")
        self.blend_type = blend_type

    def fit(self, context: 'ProblemContext'):
        """Stores the full feature matrix for later use."""
        self.all_features_ = context.df_features

    def weigh(self, combination: SliceCombination) -> pd.DataFrame:
        if not selection:
            return pd.DataFrame()

        # --- Use data stored during fit ---
        rep_features = self.all_features_.loc[selection]
        weight_results = {}

        # Loop through every original slice from the fitted data.
        for original_label, original_vec in self.all_features_.iterrows():
            # Solve the same optimization problem as before to find the
            # blend of representatives for this original_vec.
            # ... (scipy.optimize.minimize logic) ...

            # Dummy result for illustration
            dummy_blend = np.ones(len(selection)) / len(selection)
            weight_results[original_label] = dummy_blend

        blended_weights_df = pd.DataFrame.from_dict(
            weight_results,
            orient='index',
            columns=selection
        )
        return blended_weights_df

    def weigh(
            self,
            combination: SliceCombination,
            context: ProblemContext,
    ) -> pd.DataFrame:

        if not combination:
            return pd.DataFrame()

        # 1. Get feature vectors for the selection (reps) and all candidates.
        all_features = context.df_features
        rep_features = all_features.loc[combination]

        weight_results = {}

        # 2. Loop through every original slice in the full dataset.
        for original_label, original_vec in all_features.iterrows():
            # 3. For each one, solve an optimization problem to find the best blend.
            # Objective: minimize || original_vec - sum(weights * rep_vecs) ||^2
            def objective_func(weights):
                blended_vec = np.dot(weights, rep_features.values)
                return np.sum((original_vec.values - blended_vec) ** 2)

            # Initial guess: uniform weights
            initial_weights = np.ones(len(combination)) / len(combination)

            # Constraints and bounds for a convex blend
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
            bounds = [(0, 1) for _ in range(len(combination))]

            # 4. Solve for the optimal weights for this specific original_slice.
            result = minimize(
                objective_func,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            weight_results[original_label] = result.x

        # 5. Assemble the results into a final DataFrame and return.
        blended_weights_df = pd.DataFrame.from_dict(
            weight_results,
            orient='index',
            columns=combination
        )

        return blended_weights_df
