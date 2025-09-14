class SearchAlgorithm(Protocol):
    """Pillar A: The engine that finds the optimal selection."""
    def __init__(self, objective: Objective):
        self.objective = objective

    def search(self, context: ProblemContext, k: int) -> RepSetResult:
        # The core logic of the search lives here.
        # It returns a RepSetResult object populated with the selection,
        # representatives, scores, and metadata.
       ...


class CombinationGenerator(Protocol):
    """Protocol for generating candidate combinations."""
    def generate(self, candidates: List[Hashable], k: int) -> Iterator]:...

class ExhaustiveGenerator(CombinationGenerator):...

class SeasonalConstraintGenerator(CombinationGenerator):
    def __init__(self, season_map: Dict):
        self.season_map = season_map # {slice_label: season_name}

    def generate(self, candidates, k):
        # Logic to generate only combinations with at least one slice per season.
       ...

class CombinatorialSearch(SearchAlgorithm):
    def __init__(self, objective: Objective, generator: CombinationGenerator, policy: SelectionPolicy):
        self.objective = objective
        self.generator = generator
        self.policy = policy
    #...


class CombinatorialSearch(SearchAlgorithm):
    """A generate-and-test searcher that uses an external Objective."""
    def __init__(self, objective: Objective, generator: "CombinationGenerator", policy: "SelectionPolicy"):...

class ClusteringSearch(SearchAlgorithm):
    """A constructive searcher with an internal objective (e.g., minimize inertia)."""
    def __init__(self, cluster_model, selection_space: Literal['subset', 'synthetic']):...

class HullClusteringSearch(SearchAlgorithm):
    """A constructive searcher that finds extreme points."""
    def __init__(self, hull_type: Literal['convex', 'conic']):...

# --- Sub-components for CombinatorialSearch ---
class CombinationGenerator(Protocol):
    def generate(self, candidates: List[Hashable], k: int) -> Iterator]:...

class SelectionPolicy(Protocol):
    def select_best(self, evaluated_combinations: pd.DataFrame) -> Sequence[Hashable]:...
