import numpy as np

#Leitura da dimensão dos dados
n, m = map(int, input().split())

w = np.array(list(map(float, input().split())))
b = float(input())

X = np.zeros((n, m), dtype=float)
for i in range(n):
    X[i] = np.array(list(map(float, input().split())))

support_count = 0

# os support vectors estão contidos nos hiperplanos w @ x + b = +1 e w @ x + b = -1
for x in X:
    z = w @ x + b

    #Utilize a função abaixo para evitar problemas de precisão
    if (np.isclose(np.abs(z), 1.0, atol=1e-3)):
        support_count += 1

print(support_count)
