class RepresentationModel(Protocol):
    """Pillar R: Assigns weights to the final selection."""
    def assign_weights(self, result: RepSetResult, context: ProblemContext) -> None:
        # This method calculates weights and updates result.weights
       ...