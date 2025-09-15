from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Dict, Hashable

import pandas as pd

if TYPE_CHECKING:
    from ..context import ProblemContext
    from ..types import SliceCombination


class RepresentationModel(ABC):
    """
    Abstract base class for all representation models.

    The model is first fitted to a ProblemContext to learn about the entire
    dataset, and then its weigh() method can be used to calculate weights
    for any specific selection.
    """

    @abstractmethod
    def fit(self, context: 'ProblemContext'):
        """
        Fits the representation model to the full dataset context.

        This method should perform any necessary pre-computation based on the
        full set of candidates (e.g., storing the feature matrix).

        Parameters
        ----------
        context : ProblemContext
            The problem context, containing the feature space of all candidate
            slices.
        """
        ...

    @abstractmethod
    def weigh(
        self,
        combination: SliceCombination
    ) -> Union[Dict[Hashable, float], pd.DataFrame]:
        """
        Calculates the representation weights for a given selection.

        This method should be called only after the model has been fitted.

        Parameters
        ----------
        combination : SliceCombination
            The combination for which you want to receive the weighting factors for.

        Returns
        -------
        Union[Dict[Hashable, float], pd.DataFrame]
            The calculated weights.
        """
        ...
