import numpy as np

#Leitura da dimensão dos dados
n, m = map(int, input().split())

w = np.array(list(map(float, input().split())))
b = float(input())

X = np.zeros((n, m), dtype=float)
for i in range(n):
    X[i] = np.array(list(map(float, input().split())))

# classification:
# z = w @ x + b
# z > 0 => classe 1
# z < 0 => classe -1 (no caso, 0)
for x in X:
    z = w @ x + b

    if (z >= 0): print('1')
    if (z < 0): print('0')
