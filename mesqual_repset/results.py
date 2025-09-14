@dataclass
class RepSetResult:
    """The standardized output object."""
    selection: Sequence[Hashable]
    selection_space: Literal['subset', 'synthetic', 'chronological']
    representatives: pd.DataFrame # The actual data of the representatives
    scores: Dict[str, float]
    weights: Union, pd.DataFrame] = None # Populated by RepresentationModel
    metadata: Dict = field(default_factory=dict)