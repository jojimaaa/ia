from scipy import stats
import numpy as np
from jojitree import JojiTree, OrthogonalParams, GainMethod, SplitMethod, GAINMETHODS

type NDArray = np.ndarray

# Random Forest
class JojiForest:

    def __init__(
        self,
        featuresPerTree: int,
        samplesPerTree: int,
        repeatedSampling: bool = True,
        treeCount: int = 5,
        maxDepth: int = 4,
        gainMethod: GainMethod = "entropy",
        splitMethod: SplitMethod = OrthogonalParams(),
        lda_components: int | None = None,  # None = sem LDA
    ):
        if featuresPerTree <= 0:
            raise ValueError("featurePerTree must be a positive integer.")

        if not GAINMETHODS.__contains__(gainMethod):
            raise TypeError(f'Invalid Gain Method: {gainMethod}.')

        self.featuresPerTree = featuresPerTree
        self.samplesPerTree = samplesPerTree
        self.repeatedSampling = repeatedSampling
        self.treeCount = treeCount
        self.maxDepth = maxDepth
        self.gainMethod = gainMethod
        self.splitMethod = splitMethod
        self.lda_components = lda_components
        self.trees: list[JojiTree] = []

    def fit(self, X_train: NDArray, Y_train: NDArray):
        n, m = X_train.shape

        if self.samplesPerTree > n:
            raise ValueError(f"X_train must have at least {self.samplesPerTree} samples.")

        if self.featuresPerTree > m:
            raise ValueError(f"X_train must have at least {self.featuresPerTree} features.")

        rng = np.random.default_rng()
        self.trees = []

        for i in range(self.treeCount):
            baggingIndexes = rng.choice(n, size=self.samplesPerTree, replace=self.repeatedSampling)
            subsamplingCols = rng.choice(m, size=self.featuresPerTree, replace=False)

            X_bag = X_train[baggingIndexes][:, subsamplingCols]
            Y_bag = Y_train[baggingIndexes]

            tree = JojiTree(
                maxDepth=self.maxDepth,
                splitMethod=self.splitMethod,
                gainMethod=self.gainMethod
            )

            tree.fit(X_bag, Y_bag,
                     featureIndexes=subsamplingCols,
                     lda_components=self.lda_components)

            self.trees.append(tree)

    def predict(self, X_test: NDArray):
        predictions = np.zeros((self.treeCount, len(X_test)))

        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X_test)

        return stats.mode(predictions, axis=0, keepdims=True).mode.flatten()
