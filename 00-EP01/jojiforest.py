from typing import Literal
import numpy as np

type GainMethod = Literal["entropy"]
type SplitMethod = Literal["orthogonal"]
type NDArray = np.ndarray

GAINMETHODS = ['entropy']
SPLITMETHODS = ['orthogonal']

# Random Forest
class JojiForest:

    def __init__(
        self,
        featuresPerTree: int,
        treeCount: int = 5,
        maxDepth: int = 4,
        gainMethod: GainMethod  = "entropy",
        splitMethod: SplitMethod = "orthogonal"
    ):
        if (not GAINMETHODS.__contains__(gainMethod)):
            raise TypeError(f'Invalid Gain Method: {gainMethod}.')

        if (not SPLITMETHODS.__contains__(splitMethod)):
            raise TypeError(f'Invalid Split Method: {splitMethod}.')

        if (featuresPerTree <= 0):
            raise ValueError("featurePerTree must be a positive integer.")

        self.featuresPerTree = featuresPerTree
        self.treeCount = treeCount
        self.maxDepth = maxDepth
        self.gainMethod = gainMethod
        self.splitMethod = splitMethod

        return

    def _buildForest():
        return

    def fit():
        return
