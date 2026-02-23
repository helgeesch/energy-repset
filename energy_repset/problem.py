from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import ProblemContext
    from .workflow import Workflow
    from .results import RepSetResult


class RepSetExperiment:
    """Orchestrate a complete and self-contained representative subset experiment.

    This class manages the execution of a full workflow from raw data to final
    selection results. It handles feature engineering, search execution, and
    weight calculation while maintaining references to intermediate states.

    Attributes:
        raw_context: Initial ProblemContext containing raw time-series data.
        workflow: Workflow definition containing all algorithm components.
        result: Final RepSetResult after run() completes (None before execution).

    Examples:
        Run a complete experiment:

        >>> import pandas as pd
        >>> from energy_repset.problem import RepSetExperiment
        >>> from energy_repset.context import ProblemContext
        >>> from energy_repset.workflow import Workflow
        >>> from energy_repset.time_slicer import TimeSlicer
        >>> # ... (imports for feature engineer, search algo, etc.)
        >>>
        >>> # Create data and context
        >>> dates = pd.date_range('2024-01-01', periods=8760, freq='h')
        >>> df = pd.DataFrame({'demand': np.random.rand(8760)}, index=dates)
        >>> slicer = TimeSlicer(unit='month')
        >>> context = ProblemContext(df_raw=df, slicer=slicer)
        >>>
        >>> # Create workflow (see Workflow docs for details)
        >>> workflow = Workflow(
        ...     feature_engineer=feature_eng,
        ...     search_algorithm=search_algo,
        ...     representation_model=repr_model,
        ...     k=3
        ... )
        >>>
        >>> # Run experiment
        >>> experiment = RepSetExperiment(context, workflow)
        >>> result = experiment.run()
        >>> print(result.selection)  # Selected periods
        >>> print(result.weights)    # Responsibility weights
    """

    def __init__(self, context: ProblemContext, workflow: Workflow):
        """Initialize experiment with raw data context and workflow.

        Args:
            context: ProblemContext containing raw time-series data and metadata.
            workflow: Workflow defining feature engineering, search, and representation.
        """
        self.raw_context = context
        self.workflow = workflow

        # These will be populated after the run
        self._feature_context: ProblemContext = None
        self.result: RepSetResult = None

    @property
    def feature_context(self) -> ProblemContext:
        """Get the context with computed features.

        Returns:
            ProblemContext with df_features populated.

        Raises:
            ValueError: If run() or run_feature_engineer() has not been called yet.
        """
        if self._feature_context is None:
            if self.raw_context.has_features:
                self._feature_context = self.raw_context
            else:
                raise ValueError('Please call run() or run_feature_engineer() first.')
        return self._feature_context

    def run_feature_engineer(self) -> ProblemContext:
        """Run only the feature engineering step.

        This method allows you to inspect features before running the full workflow.

        Returns:
            ProblemContext with df_features populated.
        """
        self._feature_context = self.workflow.feature_engineer.run(self.raw_context)
        return self._feature_context

    def run(self) -> RepSetResult:
        """Execute the entire workflow from feature engineering to final result.

        This method orchestrates the complete selection process:
        1. Runs the feature engineer to create a new, feature-rich context
        2. Stores this feature_context for user inspection
        3. Runs the search algorithm on the feature_context
        4. Fits the representation model
        5. Calculates the final weights
        6. Stores and returns the final result

        Returns:
            RepSetResult: The selected periods, weights, scores, and diagnostics.
        """
        if self._feature_context is None and not self.raw_context.has_features:
            self.run_feature_engineer()

        feature_context = self.feature_context
        search_algorithm = self.workflow.search_algorithm
        representation_model = self.workflow.representation_model

        result = search_algorithm.find_selection(feature_context)
        if result.weights is None:
            representation_model.fit(feature_context)
            result.weights = representation_model.weigh(result.selection)

        self.result = result
        return self.result
