import numpy as np
from jojitree import JojiTree, PCAParams, OrthogonalParams
from jojiforest import JojiForest
import dataHandling as dh

def main():
    X_train, Y_train, X_test = dh.loadFile('data.npz')

    x_train, x_test, y_train, y_test = dh.splitData(X_train, Y_train)

    n_train, m = x_train.shape

    classifier = JojiTree(
        maxDepth=4,
        gainMethod='gini',
        # splitMethod=OrthogonalParams({'type': 'orthogonal'})
        splitMethod=PCAParams({'type': 'PCA', 'c': 20})
    )

    # for f in [6, 10, 15, 20, 25, 30, 34]:
    #     forest = JojiForest(
    #         featuresPerTree=f,
    #         samplesPerTree=int(n_train * 0.8),
    #         repeatedSampling=True,
    #         treeCount=20,
    #         maxDepth=3,
    #         gainMethod='gini',
    #         splitMethod=OrthogonalParams({'type': 'orthogonal'})
    #     )
    #     forest.fit(x_train, y_train)
    #     y_hatF = forest.predict(x_test)
    #     accuracy, _ = dh.accuracyAndError(y_hatF, y_test)
    #     print(f"featuresPerTree={f}: {accuracy:.4f}")

    forest = JojiForest(
        featuresPerTree=20,
        samplesPerTree= int(n_train * 0.8), # 80% das amostras
        repeatedSampling=True,
        treeCount=20,
        maxDepth=4,
        gainMethod='gini',
        # splitMethod=OrthogonalParams({'type': 'orthogonal'})
        splitMethod=PCAParams({'type': 'PCA', 'c': 20})
    )

    # train
    classifier.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    # predict
    y_hatT = classifier.predict(x_test)
    y_hatF = forest.predict(x_test)
    # y_hat = forest.trees[0].predict(x_test)

    # results = dh.crossValidate(forest, X_train, Y_train, cv=5)

    # print(y_hat)

    accuracyF, errorRateF = dh.accuracyAndError(y_hatF, y_test)
    accuracyT, errorRateT = dh.accuracyAndError(y_hatT, y_test)

    # print(f"Accuracy Forest: {accuracyF}")
    print(f"Accuracy Tree: {accuracyT}")

    return

if (__name__ == '__main__'):
    main()

# Test case
# 8 4
# 0.34 0.94 0
# 1.00 -0.09 0
# 1.32 -0.48 1
# -0.08 0.27 1
# -0.96 -0.15 0
# 1.89 0.88 1
# 0.58 -0.47 1
# 1.04 0.57 0
# -0.08 1.03
# 0.35 -0.31
# 1.89 -0.27
# -1.21 0.81
#
# Saíde: 1 0 1 0
