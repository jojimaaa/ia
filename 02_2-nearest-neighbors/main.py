import numpy as np
from scipy.stats import mode
import random

def euc_dist(x, y):
    return np.linalg.norm(x - y, ord=2)

def predict(k: int, x, X, Y):
    distances = np.array([euc_dist(x, X_i) for X_i in X])

    nearest_indices = np.argsort(distances)[:k]

    k_nearest_classes = np.array([Y[i] for i in nearest_indices])

    classification = mode(k_nearest_classes).mode

    return classification


def main():
    #Leitura das entradas
    n, m, k = map(int, input().split())

    X = np.zeros((n, m), dtype=int)
    Y = np.zeros((n, 1), dtype=int)
    x = np.zeros((1, m), dtype=int)

    for i in range(n):
        tmp = np.array(list(map(int, input().split())))
        X[i] = tmp[:-1]#Valores de X
        Y[i] = tmp[-1]#Valores de Y

    avgX = np.average(X, axis=0)
    stdX = np.std(X, axis=0)

    X_norm = (X - avgX) / stdX

    tmp = np.array(list(map(int, input().split())))
    x[0] = tmp

    x_norm = (x[0] - avgX)/stdX

    y_hat = predict(k, x_norm, X_norm, Y)

    print(y_hat[0])

main()
