import numpy as np

def getConfusionMatrixStats(M: np.ndarray) -> tuple[int, int, int]:
    N = len(M)

    if (N == 0): return -1, -1, -1
    if (N != len(M[0])): return -1, -1, -1

    total = 0
    correct = 0
    incorrect = 0

    for i in range(N):
        for j in range(N):
            if (i == j): correct += M[i][j]
            else: incorrect += M[i][j]

            total += M[i][j]

    return total, correct, incorrect

#Leitura da dimensão da matriz (quadrada)
d = int(input())

cm = np.zeros((d, d), dtype=int)

#Leitura da matriz (dxd)
for i in range(d):
    cm[i] = np.array(list(map(int, input().split())))

total, correct, incorrect = getConfusionMatrixStats(cm)

# (i) a quantidade de amostras preditas
print(total, end=' ')

#(ii) quantidade de amostras preditas corretamente
print(correct, end=' ')

#(iii) quantidade de amostras preditas incorretamente
print(incorrect)
