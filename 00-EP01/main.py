import numpy as np
from jojitree import JojiTree
from jojiforest import JojiForest
import dataHandling as dh

def main():
    X_train, Y_train, X_test = dh.loadFile('data.npz')

    x_train, x_test, y_train, y_test = dh.splitData(X_train, Y_train)

    classifier = JojiTree(maxDepth=2, gainMethod='gini', splitMethod='PCA')
    # forest = JojiForest(2, 5)


    # train
    classifier.fit(x_train, y_train)
    # forest.fit(x_train, y_train)

    # predict
    y_hat = classifier.predict(x_test)
    # y_hat = forest.predict(x_test)

    # results = dh.crossValidate(forest, X_train, Y_train, cv=5)

    accuracy, errorRate = dh.accuracyAndError(y_hat, y_test)
    print(f"Accuracy: {accuracy}")


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
