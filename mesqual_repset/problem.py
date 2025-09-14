class RepSetProblem:
    """Orchestrates the representative subset selection workflow."""
    def __init__(self, data: pd.DataFrame, slicer: TimeSlicer):
        self.context = ProblemContext(data, slicer)

    def engineer_features(self, feature_engineer: FeatureEngineer):
        """Pillar F: Creates the feature space."""
        feature_engineer.transform(self.context)
        return self

    def solve(self, search_algorithm: SearchAlgorithm, representation_model: RepresentationModel, k: int) -> RepSetResult:
        """Pillars A & R: Solves the problem and returns the final result."""
        result = search_algorithm.search(self.context, k)
        representation_model.assign_weights(result, self.context)
        return result

    def run_workflow(self, workflow: Workflow) -> RepSetResult:
        self.engineer_features(workflow.feature_engineer)
        result = self.solve(
            workflow.search_algorithm,
            workflow.representation_model,
            workflow.k
        )
        return result