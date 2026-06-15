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
        featuresPerTree: int, # subsample this many features in each tree
        samplesPerTree: int, # (p) each tree will be trained with this many samples (bagging)
        repeatedSampling: bool = True, # if bagging will be performed with repetition or not
        treeCount: int = 5, # (K) number of distinct weak learners
        maxDepth: int = 4, # max depth of each weak learner
        gainMethod: GainMethod  = "entropy",
        splitMethod: SplitMethod = "orthogonal",
    ):
        if (not GAINMETHODS.__contains__(gainMethod)):
            raise TypeError(f'Invalid Gain Method: {gainMethod}.')

        if (not SPLITMETHODS.__contains__(splitMethod)):
            raise TypeError(f'Invalid Split Method: {splitMethod}.')

        if (featuresPerTree <= 0):
            raise ValueError("featurePerTree must be a positive integer.")

        self.featuresPerTree = featuresPerTree
        self.samplesPerTree = samplesPerTree
        self.repeatedSampling = repeatedSampling
        self.treeCount = treeCount
        self.maxDepth = maxDepth
        self.gainMethod = gainMethod
        self.splitMethod = splitMethod

        return

    def _buildForest(self):
        return

    def fit(self, X_train: NDArray, Y_train: NDArray):
        n, m = X_train.shape

        # Size checks
        if (self.samplesPerTree > n):
            raise ValueError(f"X_train must have at least {self.samplesPerTree} samples.")

        if (self.featuresPerTree > m):
            raise ValueError(f"X_train must have at least f{self.featuresPerTree} features.")

        # Generate bagging + subsampling sets
        rng = np.random.default_rng()

        X_sets = []
        Y_sets = []
        n, m = X_train.shape


        for i in range(self.treeCount):
            baggingIndexes = rng.choice(
                n,
                size=self.samplesPerTree,
                replace=self.repeatedSampling
            )

            subsamplingCols = rng.choice(
                m,
                size=self.featuresPerTree,
                replace=False
            )

            X_sets.append(X_train[baggingIndexes][:, subsamplingCols])
            Y_sets.append(Y_train[baggingIndexes])

            # criar as árvores
            # cada árvore deve guardar quais (indexes) features está trabalhando

            # tree = JojiTree(...)
            # self.trees.append(tree.fit(X_sets[i], Y_sets[i], featureIndexes=subsamplingCols))

        return
