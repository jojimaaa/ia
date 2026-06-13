import numpy as np

def naiveSearch(target: np.ndarray, dataset: np.ndarray):
    for vec in dataset:
        if (np.array_equal(target, vec)): return 1
    return 0

def hasOverlap(set1: np.ndarray, set2: np.ndarray) -> bool:
    if (len(set1) == 0 | len(set2) == 0): return False

    if (len(set1[0]) != len(set2[0])): return False

    for vec1 in set1:
        if (naiveSearch(vec1, set2) == 1): return True

    return False

#Leitura das entradas
n_train, n_test, m = map(int, input().split())

trainSet = np.zeros((n_train, m))
testSet = np.zeros((n_test, m))

for i in range (n_train):
    temp = np.array(list(map(int, input().split())))
    trainSet[i] = temp

for i in range (n_test):
    temp = np.array(list(map(int, input().split())))
    testSet[i] = temp

if hasOverlap(trainSet, testSet):
    print('1')
else:
    print('0')
