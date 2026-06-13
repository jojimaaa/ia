import numpy as np
from scipy.stats import entropy

class Node():
    def __init__(self,
                 feature_i_star=None,
                 th_star=None,
                 left=None,
                 right=None,
                 label=None):

        # for decision node
        self.feature_i_star = feature_i_star
        self.th_star = th_star
        self.left = left
        self.right = right

        # y_hat for leaf nodes
        self.label = label

class DecisionTreeClassifier():
    def __init__(self):
        self.root = None

    def build_tree(self, X: np.ndarray, Y: np.ndarray):
        # Critério de parada #1: quando nesse ramo há apenas 1 classe
        if len(np.unique(Y)) == 1:
            return Node(label=Y[0]) # O Leaf Label é aquela classe

        feature_i_star, th_star = self.get_best_split(X, Y)

        # Critério de parada #2: quando não achamos uma divisão da árvore
        # Chega-se quando qualquer threshold deixaria um lado vazio
        # ou quando nenhum split gera ganho
        if feature_i_star == -1:
             return Node(label=self.calculate_leaf_label(Y))

        feature_values = X[:, feature_i_star]
        Xi_left, Yi_left = [], []
        Xi_right, Yi_right = [], []

        for i in range(len(feature_values)):
            if feature_values[i] <= th_star:
                Xi_left.append(X[i])
                Yi_left.append(Y[i])
            else:
                Xi_right.append(X[i])
                Yi_right.append(Y[i])


        # Mesmo caso do critério #2 de deixar um dos lados vazios
        if len(Xi_left) == 0 or len(Xi_right) == 0:
            return Node(label=self.calculate_leaf_label(Y))

        Xi_left = np.array(Xi_left)
        Yi_left = np.array(Yi_left)
        Xi_right = np.array(Xi_right)
        Yi_right = np.array(Yi_right)

        left_subtree = self.build_tree(Xi_left, Yi_left)
        right_subtree = self.build_tree(Xi_right, Yi_right)

        return Node(feature_i_star, th_star,
                    left_subtree, right_subtree)

    def get_best_split(self, X: np.ndarray, Y: np.ndarray):
        i_star, th_star = -1, -1
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
                    curr_info_gain = self.information_gain(Y, Yi_left, Yi_right)

                    # Busco ganho máximo
                    if curr_info_gain>max_info_gain:
                        i_star = feature_i
                        th_star = th
                        max_info_gain = curr_info_gain

        return i_star, th_star

    def entropy_calc(self, y):
        values, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return entropy(p, base=2)

    def information_gain(self, parent: np.ndarray, l_child: np.ndarray, r_child: np.ndarray):
        #***Implemente*** a função de ganho de informação
        entParent = self.entropy_calc(parent)
        entLChild = self.entropy_calc(l_child)
        entRChild = self.entropy_calc(r_child)

        nu = len(l_child) / len(parent)

        return entParent - ((nu * entLChild )+ ((1-nu)*entRChild))


    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        predictions = []
        for x in X:
            y_hat  = self.make_prediction(x, self.root)
            predictions.append(y_hat)

        return predictions

    def make_prediction(self, x, tree):

        if tree.label is not None: #Leaf
            return tree.label

        feature_val = x[tree.feature_i_star]
        if feature_val<=tree.th_star:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

def main():

    n_train, n_test = map(int, input().split())
    X_train = np.zeros((n_train, 2), dtype=float)
    Y_train = np.zeros((n_train, 1), dtype=int)
    X_test = np.zeros((n_test, 2), dtype=float)


    for i in range(n_train):
        tmp = np.array(list(map(float, input().split())))
        X_train[i] = tmp[:-1]
        Y_train[i] = int(tmp[-1])

    for i in range(n_test):
        X_test[i] = np.array(list(map(float, input().split())))


    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, Y_train)

    y_hat = classifier.predict(X_test)

    for y_ in y_hat:
        print(y_, end=' ')

main()
