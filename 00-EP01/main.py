import time
import numpy as np
from jojitree import JojiTree, PCAParams, OrthogonalParams, SVMParams
from jojiforest import JojiForest
import dataHandling as dh


def _evaluate(name, model, x_train, y_train, x_test, y_test, **fit_kwargs):
    t0 = time.perf_counter()
    model.fit(x_train, y_train, **fit_kwargs)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_hat = model.predict(x_test)
    t_pred = time.perf_counter() - t0

    acc, _ = dh.accuracyAndError(y_hat, y_test)
    print(f"{name:36s} acc={acc:.4f}  fit={t_fit:6.2f}s  pred={t_pred:5.2f}s")
    return acc


def main():
    X_train, Y_train, X_test = dh.loadFile('data.npz')
    x_train, x_test, y_train, y_test = dh.splitData(X_train, Y_train)
    n_train, m = x_train.shape

    splitMethods = {
        'orthogonal':  OrthogonalParams({'type': 'orthogonal'}),
        'SVM-linear':  SVMParams({'type': 'SVM', 'C': 1.0, 'kernel': 'linear'}),
        'SVM-rbf':     SVMParams({'type': 'SVM', 'C': 1.0, 'kernel': 'rbf'}),
        'SVM-poly':    SVMParams({'type': 'SVM', 'C': 1.0, 'kernel': 'poly'}),
        'SVM-sigmoid': SVMParams({'type': 'SVM', 'C': 1.0, 'kernel': 'sigmoid'}),
    }

    lda_options = {
        'no-LDA': None,
        'LDA-1': 1,
        'LDA-2': 2 # Máximo (#classes - 1)
    }

    print("\n" + 30 * "=" + " ÁRVORE ÚNICA " + 30 * "=")
    for lda_name, lda_c in lda_options.items():
        print(f"\n--- {lda_name} ---")
        for name, sm in splitMethods.items():
            tree = JojiTree(maxDepth=4, gainMethod='gini', splitMethod=sm)
            _evaluate(f"Tree[{name}]", tree, x_train, y_train, x_test, y_test,
                      lda_components=lda_c)

    print("\n" + 30 * "=" + " FLORESTA " + 30 * "=")
    for lda_name, lda_c in lda_options.items():
        print(f"\n--- {lda_name} ---")
        for name, sm in splitMethods.items():
            forest = JojiForest(
                featuresPerTree=20,
                samplesPerTree=int(n_train * 0.8),
                repeatedSampling=True,
                treeCount=20,
                maxDepth=4,
                gainMethod='gini',
                splitMethod=sm,
                lda_components=lda_c,
            )
            _evaluate(f"Forest[{name}]", forest, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
