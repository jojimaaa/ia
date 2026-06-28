from scipy import stats
import numpy as np
from multiprocessing import Pool
from jojitree import JojiTree, OrthogonalParams, GainMethod, SplitMethod, GAINMETHODS

type NDArray = np.ndarray

def _trainTree(args):
    X_bag, Y_bag, subsamplingCols, maxDepth, splitMethod, gainMethod, lda_components = args
    tree = JojiTree(
        maxDepth=maxDepth,
        splitMethod=splitMethod,
        gainMethod=gainMethod
    )
    tree.fit(X_bag, Y_bag, lda_components=lda_components)
    tree.originalFeatureIndexes = subsamplingCols
    return tree

def _predictTree(args):
    tree, X_test = args
    return tree.predict(X_test[:, tree.originalFeatureIndexes])

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
        lda_components: int | None = None,
        n_jobs: int = -1,  # -1 = todos os núcleos
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
        self.n_jobs = n_jobs
        self.trees: list[JojiTree] = []

    def fit(self, X_train: NDArray, Y_train: NDArray):
        n, m = X_train.shape

        if self.samplesPerTree > n:
            raise ValueError(f"X_train must have at least {self.samplesPerTree} samples.")

        if self.featuresPerTree > m:
            raise ValueError(f"X_train must have at least {self.featuresPerTree} features.")

        rng = np.random.default_rng()

        args = []
        for _ in range(self.treeCount):
            baggingIndexes = rng.choice(n, size=self.samplesPerTree, replace=self.repeatedSampling)
            subsamplingCols = rng.choice(m, size=self.featuresPerTree, replace=False)

            X_bag = X_train[baggingIndexes][:, subsamplingCols]
            Y_bag = Y_train[baggingIndexes]

            args.append((X_bag, Y_bag, subsamplingCols,
                         self.maxDepth, self.splitMethod,
                         self.gainMethod, self.lda_components))

        n_jobs = self.n_jobs if self.n_jobs > 0 else None  # None = todos os núcleos
        with Pool(n_jobs) as pool:
            self.trees = pool.map(_trainTree, args)

    def predict(self, X_test: NDArray):
        args = [(tree, X_test) for tree in self.trees]

        n_jobs = self.n_jobs if self.n_jobs > 0 else None
        with Pool(n_jobs) as pool:
            results = pool.map(_predictTree, args)

        predictions = np.array(results)  # (treeCount, n_test)
        return stats.mode(predictions, axis=0, keepdims=True).mode.flatten()
