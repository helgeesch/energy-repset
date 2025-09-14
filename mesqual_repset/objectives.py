class ScoreComponent(Protocol):
    """A single metric for the objective function."""
    name: str
    def score(self, selection: Sequence[Hashable], context: ProblemContext) -> float:...

class Objective:
    """Pillar O: A collection of ScoreComponents used by the search algorithm."""
    def __init__(self, components: List):
        self.components = components

    def evaluate(self, selection: Sequence[Hashable], context: ProblemContext) -> Dict[str, float]:
        # Returns a dictionary of scores for a given selection
       ...