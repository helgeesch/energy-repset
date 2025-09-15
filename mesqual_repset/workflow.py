from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path

if TYPE_CHECKING:
    from .feature_engineering import FeatureEngineer
    from .search_algorithms.search_algorithm import SearchAlgorithm
    from .representation import RepresentationModel


@dataclass
class Workflow:
    """A serializable object that defines a complete selection problem."""
    feature_engineer: FeatureEngineer
    search_algorithm: SearchAlgorithm
    representation_model: RepresentationModel
    k: int

    def save(self, filepath: str | Path):
        # Logic to serialize the workflow config to JSON/YAML
        pass

    @classmethod
    def load(cls, filepath: str | Path):
        # Logic to deserialize and reconstruct the workflow
        pass
