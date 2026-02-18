from typing import Literal, List, Hashable, Tuple

SliceUnit = Literal["year", "month", "week", "day", "hour"]
SliceCombination = Tuple[Hashable, ...]
ScoreComponentDirection = Literal["min", "max"]
