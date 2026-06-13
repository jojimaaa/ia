import numpy as np

#Leitura da dimensão dos dados
n, m = map(int, input().split())

#Leitura dos dados X
X = np.zeros((n, m), dtype=int)
for i in range(n):
    X[i] = np.array(list(map(int, input().split())))

#Leitura da amostra x
x = np.array(list(map(int, input().split())))

# Max X^i (maior valor da i-ésima feature)
max_i_feature = np.max(X, axis=0)

# Min X^i (menor valor da i-ésima feature)
min_i_feature = np.min(X, axis=0)

x_norm = (x - min_i_feature)/(max_i_feature - min_i_feature)

#Implemente aqui a normalização antes de exibir a amostra
for x_norm_i in x_norm.flatten():

    print(int(x_norm_i), end=' ')
