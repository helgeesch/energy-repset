@dataclass
class Workflow:
    """A serializable object that defines a complete selection problem."""
    feature_engineer: FeatureEngineer
    search_algorithm: SearchAlgorithm
    representation_model: RepresentationModel
    k: int

    def save(self, filepath: str):
        # Logic to serialize the workflow config to JSON/YAML
        pass

    @classmethod
    def load(cls, filepath: str):
        # Logic to deserialize and reconstruct the workflow
        pass
