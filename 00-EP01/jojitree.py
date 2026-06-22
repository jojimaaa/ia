from node import Node

import numpy as np
from typing import Literal, TypedDict
from scipy.stats import entropy

class OrthogonalParams(TypedDict):
    type: str = 'orthogonal'

class PCAParams(TypedDict):
    type: str = 'PCA'
    c: int



type GainMethod = Literal["entropy"]
type SplitMethod = OrthogonalParams | PCAParams
type NDArray = np.ndarray

GAINMETHODS = ['entropy', 'gini']

class JojiTree:
    def __init__(
        self,
        maxDepth: int = 4,
        splitMethod: SplitMethod = OrthogonalParams(), # random, PCA, SVM, ...
        gainMethod: GainMethod  = "gini", # entropy, ...,
        verbose: bool = False
    ):
        if splitMethod['type'] == 'PCA' and splitMethod['c'] <= 0:
            raise ValueError("c must be a positive integer.")

        if (not GAINMETHODS.__contains__(gainMethod)):
            raise TypeError(f'Invalid Gain Method: {gainMethod}.')

        self.root = None
        self.maxDepth = maxDepth
        self.gainMethod = gainMethod
        self.splitMethod = splitMethod
        self.featureIndexes = None

        if (verbose):
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
        if (X_left is None or X_right is None or Y_left is None or Y_right is None):
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

    def _informationGain(self, parent: NDArray, l_child: NDArray, r_child: NDArray) -> float:
        match self.gainMethod:
            case "entropy":

                entParent = self._entropyCalc(parent)
                entLChild = self._entropyCalc(l_child)
                entRChild = self._entropyCalc(r_child)

                nu = len(l_child) / len(parent)

                return entParent - ((nu * entLChild )+ ((1-nu)*entRChild))

            case "gini":
                giniParent = self._giniCalc(parent)
                giniLChild = self._giniCalc(l_child)
                giniRChild = self._giniCalc(r_child)

                nu = len(l_child) / len(parent)

                return giniParent - ((nu * giniLChild) + ((1-nu) * giniRChild))

        return -1

    def _entropyCalc(self, y: NDArray):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return entropy(p, base=2)

    def _giniCalc(self, y: NDArray) -> float:
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    def _calculateLeafLabel(self, Y: NDArray) -> NDArray:
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X: NDArray, Y: NDArray, featureIndexes: NDArray | None = None):
        if (featureIndexes is None):
            self.featureIndexes = np.arange(X.shape[1])
        else:
            self.featureIndexes = featureIndexes

        self.root = self._buildTree(X, Y)

    def predict(self, X: np.ndarray) -> list:
        predictions = []

        for x in X[:, self.featureIndexes]:
            y_hat  = self._makePrediction(x, self.root)
            predictions.append(y_hat)

        return predictions

    def _makePrediction(self, x: NDArray, tree: Node):
        if (self.splitMethod['type'] == 'orthogonal'):
            return self._makeOrthogonalPrediction(x, tree)
        else:
            return self._makeObliquePrediction(x, tree)

    def _makeOrthogonalPrediction(self, x: NDArray, tree: Node):
        if tree.label is not None: #Leaf
            return tree.label

        # para o caso ortogonal, w_star é o índice da feature_star
        feature_val = x[tree.w_star]
        if feature_val<=tree.th_star:
            return self._makeOrthogonalPrediction(x, tree.left)
        else:
            return self._makeOrthogonalPrediction(x, tree.right)

    def _makeObliquePrediction(self, x: NDArray, tree: Node):
        if tree.label is not None: #Leaf
            return tree.label

        if (tree.w_star @ x) <= tree.th_star:
            return self._makeObliquePrediction(x, tree.left)
        else:
            return self._makeObliquePrediction(x, tree.right)

    def _getBestSplit(self, X: np.ndarray, Y: np.ndarray):
        X_left, X_right, Y_left, Y_right, w_star, th_star = None, None, None, None, None, None
        match self.splitMethod['type']:
            case "orthogonal":
                X_left, X_right, Y_left, Y_right, w_star, th_star = self._getOrthogonalSplits(X, Y)

            case "PCA":
                X_left, X_right, Y_left, Y_right, w_star, th_star = self._getPCASplits(X, Y)

            case "SVM":
                #...
                return
        return X_left, X_right, Y_left, Y_right, w_star, th_star


    def _getOrthogonalSplits(self, X: np.ndarray, Y: np.ndarray):
        X_left_best = None
        X_right_best = None
        Y_left_best = None
        Y_right_best = None

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

    def _getPCASplits(self, X: np.ndarray, Y: np.ndarray):
        X_left_best = None
        X_right_best = None
        Y_left_best = None
        Y_right_best = None

        w_star, th_star = None, -1
        max_info_gain = -float("inf")

        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Pega os 15 primeiros componentes principais (ou menos se m < 15)
        n_components = min(self.splitMethod['c'], Vt.shape[0])
        components = Vt[:n_components]  # (n_components, m)

        for w in components:
            # Projeta X no componente principal w
            projections = X @ w  # (n,)

            thresholds = np.unique(projections)

            for th in thresholds:
                Xi_left, Yi_left = [], []
                Xi_right, Yi_right = [], []

                for i in range(len(projections)):
                    if projections[i] <= th:
                        Xi_left.append(X[i])
                        Yi_left.append(Y[i])
                    else:
                        Xi_right.append(X[i])
                        Yi_right.append(Y[i])

                if len(Xi_left) > 0 and len(Xi_right) > 0:
                    curr_info_gain = self._informationGain(Y, Yi_left, Yi_right)

                    if curr_info_gain > max_info_gain:
                        w_star = w
                        th_star = th
                        max_info_gain = curr_info_gain
                        X_left_best = Xi_left
                        X_right_best = Xi_right
                        Y_left_best = Yi_left
                        Y_right_best = Yi_right

        return X_left_best, X_right_best, Y_left_best, Y_right_best, w_star, th_star
