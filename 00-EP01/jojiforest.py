from scipy import stats
import numpy as np
from jojitree import JojiTree, OrthogonalParams, GainMethod, SplitMethod, GAINMETHODS

type NDArray = np.ndarray

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
        splitMethod: SplitMethod = OrthogonalParams(),
    ):
        if (featuresPerTree <= 0):
            raise ValueError("featurePerTree must be a positive integer.")

        if (not GAINMETHODS.__contains__(gainMethod)):
            raise TypeError(f'Invalid Gain Method: {gainMethod}.')

        self.featuresPerTree = featuresPerTree
        self.samplesPerTree = samplesPerTree
        self.repeatedSampling = repeatedSampling
        self.treeCount = treeCount
        self.maxDepth = maxDepth
        self.gainMethod = gainMethod
        self.splitMethod = splitMethod
        self.trees: list[JojiTree] = []
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

            tree = JojiTree(
                maxDepth=self.maxDepth,
                splitMethod=self.splitMethod,
                gainMethod=self.gainMethod
            )

            tree.fit(X_sets[i], Y_sets[i], featureIndexes=subsamplingCols)

            self.trees.append(tree)
        return

    def predict(self, X_test: NDArray):
        predictions = np.zeros((self.treeCount, len(X_test)))
        y_hat = np.zeros(len(X_test))

        for i, tree in enumerate(self.trees):
            # print(40*"=")
            # print(f'predicting tree {i}')
            result = tree.predict(X_test)
            predictions[i] = result

        y_hat = stats.mode(predictions, axis=0, keepdims=True)


        return y_hat.mode.flatten()
