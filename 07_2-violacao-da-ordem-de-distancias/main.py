import numpy as np
import random
from itertools import permutations
def main():

    s, n, m, c = map(int, input().split())

    np.random.seed(s)
    random.seed(s)

    X = np.random.randn(n, m)
    X = X - X.mean(axis=0)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # PCA - X projetada nos primeiros c componentes base V
    # Vt -> Ortonormal, cada linha é um vetor da base
    # Vt.T -> Ortonormal, cada coluna é um vetor da base
    # Mudança de Base x @ Mb, onde x é o vetor e
    # Mb é a matriz de mudança de base com os vetores nas colunas
    Xh = X @ Vt[:c].T

    count = 0

    # Devo permutar 3 a 3 sobre todos os n samples
    for i, j, k in permutations(range(n), 3):
        xi, xj, xk = X[i], X[j], X[k]
        xhi, xhj, xhk = Xh[i], Xh[j], Xh[k]

        dij = np.linalg.norm(xi - xj, ord=2)
        dik = np.linalg.norm(xi - xk, ord=2)

        cond = dij < dik

        dhij = np.linalg.norm(xhi - xhj, ord=2)
        dhik = np.linalg.norm(xhi - xhk, ord=2)

        condh = dhij < dhik

        if cond != condh:
            count = count + 1

    print(count)

main()
