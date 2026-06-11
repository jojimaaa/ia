from node import Node

import numpy as np
from typing import Literal
from scipy.stats import entropy

type GainMethod = Literal["entropy"]
type SplitMethod = Literal["orthogonal"]
type NDArray = np.ndarray

GAINMETHODS = ['entropy']
SPLITMETHODS = ['orthogonal']

class JojiTree:
    def __init__(
        self,
        maxDepth: int = 4,
        gainMethod: GainMethod  = "entropy",
        splitMethod: SplitMethod = "orthogonal"
    ):
        if (not GAINMETHODS.__contains__(gainMethod)):
            raise TypeError(f'Invalid Gain Method: {gainMethod}.')

        if (not SPLITMETHODS.__contains__(splitMethod)):
            raise TypeError(f'Invalid Split Method: {splitMethod}.')

        self.root = None
        self.maxDepth = maxDepth
        self.gainMethod = gainMethod
        self.splitMethod = splitMethod

        print(18*"=" + f" Created JojiTree " + 18*"=")
        print(f'gainMethod: {self.gainMethod}')
        print(f'splitMethod: {self.splitMethod}')
        print(f'maxDepth: {self.maxDepth}')

    def _buildTree(self, X: NDArray, Y: NDArray, depth: int = 0) -> Node:
        # Critério de parada MaxDepth
        if depth >= self.maxDepth:
            return Node(label=self._calculateLeafLabel(Y))

        # Critério de parada #1: quando nesse ramo há apenas 1 classe
        if len(np.unique(Y)) == 1:
            return Node(label=Y[0]) # O Leaf Label é aquela classe

        X_left, X_right, Y_left, Y_right, w_star, th_star = self._getBestSplit(X, Y)

        # Critério de parada #2: quando não achamos uma divisão da árvore
        # Chega-se quando qualquer threshold deixaria um lado vazio
        # ou quando nenhum split gera ganho
        if (X_left == None or X_right == None or Y_left == None or Y_right == None):
             return Node(label=self._calculateLeafLabel(Y))

        if len(X_left) == 0 or len(X_right) == 0:
            return Node(label=self._calculateLeafLabel(Y))

        Xi_left = np.array(X_left)
        Yi_left = np.array(Y_left)
        Xi_right = np.array(X_right)
        Yi_right = np.array(Y_right)

        left_subtree = self._buildTree(Xi_left, Yi_left, depth=depth+1)
        right_subtree = self._buildTree(Xi_right, Yi_right, depth=depth+1)

        return Node(w_star, th_star,
                    left_subtree, right_subtree)

    def _informationGain(self, parent: NDArray, l_child: NDArray, r_child: NDArray) -> NDArray:
        match self.gainMethod:
            case "entropy":

                entParent = self._entropyCalc(parent)
                entLChild = self._entropyCalc(l_child)
                entRChild = self._entropyCalc(r_child)

                nu = len(l_child) / len(parent)

                return entParent - ((nu * entLChild )+ ((1-nu)*entRChild))

            case "gini":
                return -1


        return -1

    def _entropyCalc(self, y: NDArray):
        values, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return entropy(p, base=2)

    def _calculateLeafLabel(self, Y: NDArray) -> NDArray:
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X: NDArray, Y: NDArray):
        self.root = self._buildTree(X, Y)

    def predict(self, X: np.ndarray) -> list:
        predictions = []
        for x in X:
            y_hat  = self._makePrediction(x, self.root)
            predictions.append(y_hat)

        return predictions

    def _makePrediction(self, x: NDArray, tree: Node):
        match self.splitMethod:
            case "orthogonal":
                return self._makeOrthogonalPrediction(x, tree)

    def _getBestSplit(self, X: np.ndarray, Y: np.ndarray):
        X_left, X_right, Y_left, Y_right, w_star, th_star = None, None, None, None, None, None
        match self.splitMethod:
            case "orthogonal":
                X_left, X_right, Y_left, Y_right, w_star, th_star = self._getOrthogonalSplits(X, Y)

        return X_left, X_right, Y_left, Y_right, w_star, th_star

    def _makeOrthogonalPrediction(self, x: NDArray, tree: Node):
        if tree.label is not None: #Leaf
            return tree.label

        feature_val = x[tree.w_star]
        if feature_val<=tree.th_star:
            return self._makeOrthogonalPrediction(x, tree.left)
        else:
            return self._makeOrthogonalPrediction(x, tree.right)

    def _getOrthogonalSplits(self, X: np.ndarray, Y: np.ndarray):
        X_left_best = None
        X_right_best = None
        X_left_best = None
        X_right_best = None

        w_star, th_star = -1, -1
        max_info_gain = -float("inf")

        # Número de colunas (features) de X
        # equivalente a
        # _, m = X.shape
        m = X.shape[-1]

        for feature_i in range(m):
            feature_values = X[:, feature_i] # extrai a (feature-i + 1)-ésima coluna de X

            # Em vez de pegar o eixo feature_i e sair varrendo com incrementos pequenos,
            # podemos apenas pegar todos os valores únicos do eixo pois os valores
            # intermediários entre 2 consecutivos seriam cálculos repetidos ou sub-optimos
            thresholds = np.unique(feature_values)

            for th in thresholds:
                Xi_left, Yi_left = [], []
                Xi_right, Yi_right = [], []

                # Varrendo todos os elementos da coluna feature_i
                for i in range(len(feature_values)):

                    # Quem for menor ou igual ao th vai para a esquerda
                    if feature_values[i] <= th:
                        Xi_left.append(X[i])
                        Yi_left.append(Y[i])
                    # Caso contrário, vai para a direita
                    else:
                        Xi_right.append(X[i])
                        Yi_right.append(Y[i])

                if len(Xi_left)>0 and len(Xi_right)>0:
                    curr_info_gain = self._informationGain(Y, Yi_left, Yi_right)

                    # Busco ganho máximo
                    if curr_info_gain>max_info_gain:
                        w_star = feature_i # Plano na feature_i
                        th_star = th
                        max_info_gain = curr_info_gain
                        X_left_best = Xi_left
                        X_right_best = Xi_right
                        Y_left_best = Yi_left
                        Y_right_best = Yi_right

        return X_left_best, X_right_best, Y_left_best, Y_right_best, w_star, th_star
