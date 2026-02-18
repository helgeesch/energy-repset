"""Result diagnostics for analyzing selection outcomes.

This module provides diagnostic tools for visualizing the results of the
search algorithm and representation model, including responsibility weights,
score component contributions, and Pareto front exploration.
"""

from .responsibility_bars import ResponsibilityBars
from .pareto_scatter import ParetoScatter2D, ParetoScatterMatrix
from .pareto_parallel_coords import ParetoParallelCoordinates
from .score_contribution_bars import ScoreContributionBars

__all__ = [
    'ResponsibilityBars',
    'ParetoScatter2D',
    'ParetoScatterMatrix',
    'ParetoParallelCoordinates',
    'ScoreContributionBars',
]
