import numpy as np
import random

def main():

    #Leitura das entradas
    s, n, m = map(int, input().split())
    v = float(input())

    if (v < 0 or v > 1):
        print('v must be in the [0, 1] interval.')
        return

    #Não altere
    np.random.seed(s)
    random.seed(s)

    X = np.random.randn(n, m)
    Xc = X - X.mean(axis=0) # dados normalizados

    U, Sigma, Vt = np.linalg.svd(Xc, full_matrices=False)

    variancia = Sigma ** 2
    variancia_total = variancia.sum()

    var_proporcoes = variancia / variancia_total

    var_acc = 0
    for index, prop in enumerate(var_proporcoes):
        var_acc += prop
        if (var_acc >= v):
            print(index+1)
            return

    print(-1)
    return

if (__name__ == '__main__'):
    main()
