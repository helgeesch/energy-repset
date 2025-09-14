from dataclasses import dataclass
from typing import List, Dict, Hashable

import pandas as pd


@dataclass
class ProblemContext:
    """A data container passed through the entire workflow."""
    df_raw: pd.DataFrame
    slicer: TimeSlicer
    candidates: List[Hashable]
    df_features: pd.DataFrame = None
    variable_weights: Dict = field(default_factory=dict)
    feature_weights: Dict = field(default_factory=dict)

    def get_sliced_data(self) -> Dict:
        """Generates sliced raw data on demand."""
        #... implementation...
        pass
