"""Result diagnostics for analyzing selection outcomes.

This module provides diagnostic tools for visualizing the results of the
search algorithm and representation model, including responsibility weights
and score component contributions.
"""

from .responsibility_bars import ResponsibilityBars

__all__ = [
    'ResponsibilityBars',
]
