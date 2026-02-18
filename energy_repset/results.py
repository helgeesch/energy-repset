from typing import Sequence, Hashable, Literal, Dict, Union, Any
from dataclasses import dataclass, field

import pandas as pd

from .types import SliceCombination
from .context import ProblemContext

@dataclass
class RepSetResult:
    """The standardized output object."""
    context: ProblemContext
    selection_space: Literal['subset', 'synthetic', 'chronological']
    selection: SliceCombination
    scores: Dict[str, float]
    representatives: Dict[Hashable, pd.DataFrame]  # The actual data of the representatives
    weights: Union[Dict[Hashable, float], pd.DataFrame] = None  # Populated by RepresentationModel
    diagnostics: Dict[str, Any] = field(default_factory=dict)
