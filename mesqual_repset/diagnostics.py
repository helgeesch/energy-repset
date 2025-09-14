class ResultAnalyzer:
    def __init__(self, result: RepSetResult):
        self.result = result

    def plot_pareto_front(self):
        # Assumes the result came from a Pareto search and has the
        # necessary data in its metadata.
        pass

    def plot_feature_space(self, context: ProblemContext):
        # Plots all candidates and highlights the selected ones.
        pass