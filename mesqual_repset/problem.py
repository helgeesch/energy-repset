from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import ProblemContext
    from .workflow import Workflow
    from .results import RepSetResult


class RepSetExperiment:
    """Orchestrates a complete and self-contained representative subset experiment."""

    def __init__(self, context: ProblemContext, workflow: Workflow):
        """
        Initializes the experiment with raw data context and a full workflow.
        """
        self.raw_context = context
        self.workflow = workflow

        # These will be populated after the run
        self._feature_context: ProblemContext = None
        self.result: RepSetResult = None

    @property
    def feature_context(self) -> ProblemContext:
        if self._feature_context is None:
            if self.raw_context._df_features is not None:
                self._feature_context = self.raw_context.copy()
            else:
                raise ValueError('Please call run() or run_feature_engineer() first.')
        return self._feature_context

    def run_feature_engineer(self) -> ProblemContext:
        self._feature_context = self.workflow.feature_engineer.run(self.raw_context)
        return self._feature_context

    def run(self):
        """
        Executes the entire workflow from feature engineering to final result.

        1. Runs the feature engineer to create a new, feature-rich context.
        2. Stores this feature_context for user inspection.
        3. Runs the search algorithm on the feature_context.
        4. Calculates the final weights.
        5. Stores and returns the final result.
        """
        if (self._feature_context is None) and (self.raw_context._df_features is None):
            self.run_feature_engineer()

        feature_context = self.feature_context
        search_algorithm = self.workflow.search_algorithm
        representation_model = self.workflow.representation_model
        k = self.workflow.k

        result = search_algorithm.find_selection(feature_context, k)
        representation_model.fit(feature_context)
        weights = representation_model.weigh(result.selection)
        result.weights = weights

        self.result = result
        return self.result
