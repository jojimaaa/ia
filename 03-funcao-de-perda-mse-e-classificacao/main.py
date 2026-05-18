#Leitura das entradas
t, n = map(int, input().split())

def calculateSquareDiff(y_hat, y_true):
    squareDiff = ((y_hat - y_true) ** 2)
    return squareDiff

value = 0

#Leitura dos valores preditos e ground-truth
for _ in range(n):
    y_hat, y_true = map(int, input().split())
    match t:
        case 0: #regressão
          value += calculateSquareDiff(y_hat, y_true)
        case 1: #classificação
          if (y_hat == y_true):
             value += 1

if (t == 0):
  value /= n

print(int(value))
