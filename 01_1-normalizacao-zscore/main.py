import numpy as np

#Leitura da dimensão dos dados
n, m = map(int, input().split())

#Leitura dos dados X
X = np.zeros((n, m), dtype=int)
for i in range(n):
    X[i] = np.array(list(map(int, input().split())))

#Leitura da amostra x
x = np.array(list(map(int, input().split())))

#Implemente aqui a normalização antes de exibir a amostra
media = np.mean(X, axis=0) # axis=0 calcula sobre as colunas de uma array 2D
desvio = np.std(X, axis=0)

norm_x = (x - media) / desvio

for norm_x_i in norm_x.flatten():
    print(int(norm_x_i), end=' ')
