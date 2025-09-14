from abc import Protocol

class FeatureEngineer(Protocol):
    """Pillar F: Transforms raw sliced data into a feature matrix."""
    def transform(self, context: ProblemContext) -> None:
        # This method calculates features and updates context.features
       ...

class FeaturePipeline(FeatureEngineer):
    def __init__(self, engineers: List[FeatureEngineer]):
        self.engineers = engineers

    def transform(self, context: ProblemContext) -> None:
        all_features =
        for engineer in self.engineers:
            # Each engineer works on a fresh copy of the context
            # to avoid side-effects, and returns its feature df.
            features = engineer.transform_and_get_features(context)
            all_features.append(features)

        # Concatenate all generated features
        context.features = pd.concat(all_features, axis=1)
