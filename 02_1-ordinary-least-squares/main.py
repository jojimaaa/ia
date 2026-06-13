import numpy as np
from scipy import linalg

def __main__():
    n, m = map(int, input().split())

    X = np.zeros((n, m), dtype=int) # n x m
    Y = np.zeros((n, 1), dtype=int) # n x 1
    x = np.zeros((1, m), dtype=int) # 1 x m

    for i in range(n):
        tmp = np.array(list(map(int, input().split())))
        X[i] = tmp[:-1]
        Y[i] = tmp[-1]

    # Normalização z-score

    avgX = np.average(X, axis=0)
    stdX = np.std(X, axis=0)
    avgY = np.average(Y, axis=0)
    stdY = np.std(Y, axis=0)

    X_norm = (X - avgX) / stdX
    Y_norm = (Y - avgY) / stdY

    # Cálculo da PseudoInversa

    X_pinv = linalg.pinv(X_norm) # m x n

    W_star = X_pinv @ Y_norm # m x 1

    tmp = np.array(list(map(int, input().split())))
    x[0] = tmp

    x_norm = (x[0] - avgX)/stdX

    W_star_T = np.transpose(W_star) # 1 x m

    # Predição por produto escalar (apenas 1)

    y_hat_norm = np.dot(x_norm, W_star_T[0])

    y_hat = (y_hat_norm * stdY[0]) + avgY[0]

    print(int(y_hat))

__main__()
