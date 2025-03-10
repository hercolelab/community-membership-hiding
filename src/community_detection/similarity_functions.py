from typing import List, Callable
from src.utils.utils import SimilarityFunctionsNames
import igraph as ig
import numpy as np

class CommunitySimilarity:
    """Class to compute the similarity between two lists of integers"""

    def __init__(self, function_name: str) -> None:
        self.function_name = getattr(SimilarityFunctionsNames, function_name).value

    def select_similarity_function(self) -> Callable[[List[int], List[int]], float]:
        """
        Select the similarity function to use

        Returns
        -------
        Callable
            Similarity function to use
        """
        if self.function_name == SimilarityFunctionsNames.SOR.value:
            return self.sorensen_similarity
        else:
            raise Exception("Similarity function not found")

    @staticmethod
    def sorensen_similarity(a: List[int], b: List[int]) -> float:
        """
        Compute the Sorensen similarity between two lists, A and B:
            S(A,B) = 2 * |A ∩ B| / (|A| + |B|)

        Parameters
        ----------
        a : List[int]
            First List
        b : List[int]
            Second List

        Returns
        -------
        float
            Sorensen similarity between the two lists, between 0 and 1
        """
        assert len(a) > 0 # List A must be not empty
        if len(b) == 0:
            return 0

        # Convert lists to sets
        a_set = set(a)
        b_set = set(b)
        # Compute the intersection
        intersection = a_set.intersection(b_set)
        return 2 * len(intersection) / (len(a_set) + len(b_set))