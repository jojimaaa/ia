"""
PCS 3838 - Inteligência Artificial - Trabalho Prático 1
Oblique Random Forest (oRF) — implementação em arquivo único.

Os hiperplanos são definidos POR NÓ (sem pré-projeção do conjunto de dados):
  - 'orthogonal' : split axis-aligned (baseline ortogonal)
  - 'PCA'        : melhor componente principal calculado com o X do próprio nó
  - 'SVM'        : hiperplano de um SVM (linear ou kernel) treinado no próprio nó
"""

import os
import time
import warnings
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# ---- MÉTODO DE SPLIT (escolha UM; deixe só uma linha sem comentário) --------
# SPLIT_METHOD = {'type': 'orthogonal'}                               # baseline ortogonal (eixo-a-eixo)
# SPLIT_METHOD = {'type': 'PCA', 'c': 5}                            # 'c' = nº de componentes principais testados por nó
SPLIT_METHOD = {'type': 'SVM', 'C': 0.0402, 'kernel': 'linear'}      # kernel: 'linear' | 'rbf' | 'poly' | 'sigmoid'

# ---- MEDIDA DE IMPUREZA -----------------------------------------------------
GAIN_METHOD = 'entropy'          # 'gini' | 'entropy'

# ---- CRITÉRIO DE PARADA DA ÁRVORE -------------------------------------------
MAX_DEPTH = 9                 # profundidade máxima de cada árvore (melhor no desempate)

# ---- LDA OPCIONAL (projeção supervisionada antes do split) ------------------
LDA_COMPONENTS = None         # None (desliga) | 1 | 2 ... (máx = nº de classes - 1)

# ---- PARÂMETROS DA FLORESTA (oRF) -------------------------------------------
TREE_COUNT = 162              # nº de árvores (mais árvores = predição mais estável)
FEATURES_PER_TREE = 23        # nº de features sorteadas por árvore (subespaço aleatório)
SAMPLES_FRACTION = 0.71        # fração do treino usada por árvore (bagging)
REPEATED_SAMPLING = True      # amostragem com reposição (bootstrap)
N_JOBS = -1                   # núcleos no treino/predição (-1 = todos)

# ---- SPLIT TREINO/VALIDAÇÃO (para medir acurácia localmente) ----------------
TEST_SIZE = 0.25              # fração separada para validação
RANDOM_STATE = 42             # semente (reprodutibilidade)

# ---- O QUE RODAR ------------------------------------------------------------
RUN_SINGLE_TREE = True        # avaliar uma árvore única
RUN_FOREST = True             # avaliar a floresta

# ---- SUBMISSÃO KAGGLE -------------------------------------------------------
GENERATE_SUBMISSION = True    # treina no X_train inteiro, prevê o X_test e gera o CSV
SUBMISSION_DIR = 'outputs'    # pasta onde o CSV é salvo (criada se não existir)


# =============================================================================
# Node
# =============================================================================
class Node():
    def __init__(self,
                 w_star=None,       # vetor normal (ortogonal/PCA/SVM linear)
                 th_star=None,
                 left=None,
                 right=None,
                 label=None,
                 svm_model=None,    # objeto SVC (SVM com kernel)
                 scaler=None):      # StandardScaler do nó

        # for decision node
        self.w_star = w_star
        self.th_star = th_star
        self.left = left
        self.right = right

        # y_hat for leaf nodes
        self.label = label

        # para SVM com kernel
        self.svm_model = svm_model
        self.scaler = scaler


# =============================================================================
# JojiTree (Oblique Decision Tree)
# =============================================================================
class OrthogonalParams(TypedDict):
    type: str = 'orthogonal'

class PCAParams(TypedDict):
    type: str = 'PCA'
    c: int

class SVMParams(TypedDict):
    type: str = 'SVM'
    C: float
    kernel: str  # 'linear', 'rbf', 'poly', 'sigmoid'


type GainMethod = Literal["entropy"]
type SplitMethod = OrthogonalParams | PCAParams | SVMParams
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
        self.originalFeatureIndexes = None

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

        X_left, X_right, Y_left, Y_right, w_star, th_star, svm_model, scaler = self._getBestSplit(X, Y)

        if (X_left is None or X_right is None or Y_left is None or Y_right is None):
            return Node(label=self._calculateLeafLabel(Y))

        if len(X_left) == 0 or len(X_right) == 0:
            return Node(label=self._calculateLeafLabel(Y))

        Xi_left, Yi_left = np.array(X_left), np.array(Y_left)
        Xi_right, Yi_right = np.array(X_right), np.array(Y_right)

        left_subtree = self._buildTree(Xi_left, Yi_left, depth=depth+1)
        right_subtree = self._buildTree(Xi_right, Yi_right, depth=depth+1)

        return Node(w_star, th_star, left_subtree, right_subtree,
                    svm_model=svm_model, scaler=scaler)

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

    def fit(self, X: NDArray, Y: NDArray, lda_components: int | None = None):

        self.lda = None
        if lda_components is not None:
            n_classes = len(np.unique(Y))
            n_components = min(lda_components, n_classes - 1)
            self.lda = LinearDiscriminantAnalysis(n_components=n_components)
            X = self.lda.fit_transform(X, Y.ravel())

        self.root = self._buildTree(X, Y)

    def predict(self, X: np.ndarray) -> list:
        if self.lda is not None:
            X = self.lda.transform(X)

        predictions = []
        for x in X:
            y_hat = self._makePrediction(x, self.root)
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
        if tree.label is not None:
            return tree.label

        if tree.svm_model is not None:
            # SVM com kernel: usa decision_function
            x_scaled = tree.scaler.transform(x.reshape(1, -1))
            projection = tree.svm_model.decision_function(x_scaled)[0]
        else:
            # PCA ou SVM linear: produto interno direto
            projection = tree.w_star @ x

        if projection <= tree.th_star:
            return self._makeObliquePrediction(x, tree.left)
        else:
            return self._makeObliquePrediction(x, tree.right)

    def _getBestSplit(self, X, Y):
        svm_model, scaler = None, None
        match self.splitMethod['type']:
            case "orthogonal":
                X_left, X_right, Y_left, Y_right, w_star, th_star = self._getOrthogonalSplits(X, Y)
            case "PCA":
                X_left, X_right, Y_left, Y_right, w_star, th_star = self._getPCASplits(X, Y)
            case "SVM":
                X_left, X_right, Y_left, Y_right, w_star, th_star, svm_model, scaler = self._getSVMSplits(X, Y)

        return X_left, X_right, Y_left, Y_right, w_star, th_star, svm_model, scaler


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

        local_mean = X.mean(axis=0)

        Xc = X - local_mean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Pega os 'c' primeiros componentes principais (ou menos se m < 'c')
        n_components = min(self.splitMethod['c'], Vt.shape[0])

        X_proj = U[:, :n_components] * S[:n_components]

        for index, projections in enumerate(X_proj.T):
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


                        th_star_adjusted = th + (local_mean @ Vt[index])
                        w_star = Vt[index]
                        th_star = th_star_adjusted
                        max_info_gain = curr_info_gain
                        X_left_best = Xi_left
                        X_right_best = Xi_right
                        Y_left_best = Yi_left
                        Y_right_best = Yi_right

        return X_left_best, X_right_best, Y_left_best, Y_right_best, w_star, th_star

    def _bestThresholdForProjection(self, projections: NDArray, Y: NDArray):
        """Dado z = X @ w, encontra o threshold que maximiza o ganho.

        Vetorizado: ordena uma vez e varre apenas os pontos médios entre
        valores consecutivos (candidatos ótimos), sem montar listas por amostra.
        Retorna (best_th, best_gain) ou (None, -inf) se nenhum split válido.
        """
        order = np.argsort(projections, kind="mergesort")
        z = projections[order]

        # Candidatos: pontos médios onde z muda de valor (split válido com ambos os lados não-vazios)
        change = np.where(z[1:] != z[:-1])[0]
        if change.size == 0:
            return None, -float("inf")

        thresholds = (z[change] + z[change + 1]) / 2.0

        best_th, best_gain = None, -float("inf")
        for th in thresholds:
            mask = projections <= th
            gain = self._informationGain(Y, Y[mask], Y[~mask])
            if gain > best_gain:
                best_gain = gain
                best_th = th

        return best_th, best_gain

    def _splitByProjection(self, X: NDArray, Y: NDArray, w: NDArray, th: float):
        z = X @ w
        mask = z <= th
        return X[mask], X[~mask], Y[mask], Y[~mask]

    def _getSVMSplits(self, X: NDArray, Y: NDArray):
        classes = np.unique(Y)
        C = self.splitMethod.get('C', 1.0)
        kernel = self.splitMethod.get('kernel', 'linear')

        X_left_best = X_right_best = Y_left_best = Y_right_best = None
        w_star, th_star = None, -1
        svm_best, scaler_best = None, None
        max_info_gain = -float("inf")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        partitions = []
        k = len(classes)
        for r in range(1, k // 2 + 1):
            for group in combinations(classes, r):
                if r == k - r and classes[0] not in group:
                    continue
                partitions.append(set(group))

        for group in partitions:
            binary_y = np.array([0 if c in group else 1 for c in Y])

            if len(np.unique(binary_y)) < 2:
                continue

            try:
                if kernel == 'linear':
                    svm = LinearSVC(C=C, dual='auto', max_iter=2000)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=ConvergenceWarning)
                        svm.fit(X_scaled, binary_y)
                    projections = svm.decision_function(X_scaled)
                else:
                    svm = SVC(kernel=kernel, C=C, max_iter=-1)
                    svm.fit(X_scaled, binary_y)
                    projections = svm.decision_function(X_scaled)
            except Exception:
                continue

            th, gain = self._bestThresholdForProjection(projections, Y)
            if th is None:
                continue

            if gain > max_info_gain:
                max_info_gain = gain
                th_star = th

                mask = projections <= th
                X_left_best = X[mask]
                X_right_best = X[~mask]
                Y_left_best = Y[mask]
                Y_right_best = Y[~mask]

                if kernel == 'linear':
                    # LinearSVC treinou em X escalado, e 'th' foi escolhido contra
                    # decision_function(x_scaled) = w·x - w·média + intercept.
                    # Para o predict (w_star @ x_cru <= th_star) reproduzir a MESMA
                    # partição, dobramos a média do scaler e o intercept no threshold
                    # e NÃO normalizamos w_star (normalizar muda a escala do th).
                    w = svm.coef_[0] / scaler.scale_
                    w_star = w
                    th_star = th + w @ scaler.mean_ - svm.intercept_[0]
                    svm_best = None
                    scaler_best = None
                else:
                    # Kernel não-linear: guarda modelo e scaler para predição
                    w_star = None
                    svm_best = svm
                    scaler_best = scaler

        return X_left_best, X_right_best, Y_left_best, Y_right_best, w_star, th_star, svm_best, scaler_best


# =============================================================================
# JojiForest (Oblique Random Forest)
# =============================================================================
def _trainTree(args):
    X_bag, Y_bag, subsamplingCols, maxDepth, splitMethod, gainMethod, lda_components = args
    tree = JojiTree(
        maxDepth=maxDepth,
        splitMethod=splitMethod,
        gainMethod=gainMethod
    )
    tree.fit(X_bag, Y_bag, lda_components=lda_components)
    tree.originalFeatureIndexes = subsamplingCols
    return tree

def _predictTree(args):
    tree, X_test = args
    return tree.predict(X_test[:, tree.originalFeatureIndexes])

# Random Forest
class JojiForest:

    def __init__(
        self,
        featuresPerTree: int,
        samplesPerTree: int,
        repeatedSampling: bool = True,
        treeCount: int = 5,
        maxDepth: int = 4,
        gainMethod: GainMethod = "entropy",
        splitMethod: SplitMethod = OrthogonalParams(),
        lda_components: int | None = None,
        n_jobs: int = -1,  # -1 = todos os núcleos
    ):
        if featuresPerTree <= 0:
            raise ValueError("featurePerTree must be a positive integer.")

        if not GAINMETHODS.__contains__(gainMethod):
            raise TypeError(f'Invalid Gain Method: {gainMethod}.')

        self.featuresPerTree = featuresPerTree
        self.samplesPerTree = samplesPerTree
        self.repeatedSampling = repeatedSampling
        self.treeCount = treeCount
        self.maxDepth = maxDepth
        self.gainMethod = gainMethod
        self.splitMethod = splitMethod
        self.lda_components = lda_components
        self.n_jobs = n_jobs
        self.trees: list[JojiTree] = []

    def fit(self, X_train: NDArray, Y_train: NDArray):
        n, m = X_train.shape

        if self.samplesPerTree > n:
            raise ValueError(f"X_train must have at least {self.samplesPerTree} samples.")

        if self.featuresPerTree > m:
            raise ValueError(f"X_train must have at least {self.featuresPerTree} features.")

        rng = np.random.default_rng()

        args = []
        for _ in range(self.treeCount):
            baggingIndexes = rng.choice(n, size=self.samplesPerTree, replace=self.repeatedSampling)
            subsamplingCols = rng.choice(m, size=self.featuresPerTree, replace=False)

            X_bag = X_train[baggingIndexes][:, subsamplingCols]
            Y_bag = Y_train[baggingIndexes]

            args.append((X_bag, Y_bag, subsamplingCols,
                         self.maxDepth, self.splitMethod,
                         self.gainMethod, self.lda_components))

        n_jobs = self.n_jobs if self.n_jobs > 0 else None  # None = todos os núcleos
        with Pool(n_jobs) as pool:
            self.trees = pool.map(_trainTree, args)

    def predict(self, X_test: NDArray):
        args = [(tree, X_test) for tree in self.trees]

        n_jobs = self.n_jobs if self.n_jobs > 0 else None
        with Pool(n_jobs) as pool:
            results = pool.map(_predictTree, args)

        predictions = np.array(results)  # (treeCount, n_test)
        return stats.mode(predictions, axis=0, keepdims=True).mode.flatten()


# =============================================================================
# Data handling
# =============================================================================
def loadFile(path: str):
    data = np.load(path)

    X_train = data['X_train']
    Y_train = data['y_train']
    X_test = data['X_test']


    print(25*"=" + f"\nLoaded {path}")
    return X_train, Y_train, X_test

def splitData(
    X: np.ndarray,
    Y: np.ndarray,
    testSize: float = 0.25,
    randomState: int = 42,
    shuffle: bool = True
):
    # estratifica para preservar a proporção das 3 classes (desbalanceadas)
    # no train e no test locais — só faz sentido com shuffle=True
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=testSize,
        random_state=randomState,
        shuffle=shuffle,
        stratify=Y if shuffle else None
    )
    return X_train, X_test, y_train, y_test

def exportPredictions(
    y_pred: np.ndarray,
    outDir: str = 'outputs'
) -> str:
    dataframe = pd.DataFrame({
        'ID': np.arange(1, len(y_pred) + 1),
        'Prediction': y_pred
    })

    os.makedirs(outDir, exist_ok=True)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')  # sem ':' (válido no Windows)
    path = os.path.join(outDir, f"predictions_{now}.csv")

    dataframe.to_csv(path, index=False)
    return path

def accuracyAndError(
    y_hat: np.ndarray, # predictions
    y_exp: np.ndarray  # expected
):
    # calculating error rate
    hits = 0

    for index, y in enumerate(y_hat):
        if (y == y_exp[index]):
            hits += 1

    accuracy = hits / len(y_exp)
    errorRate = 1 - accuracy

    return accuracy, errorRate


# =============================================================================
# main
# =============================================================================
def _evaluate(name, model, x_train, y_train, x_test, y_test, **fit_kwargs):
    t0 = time.perf_counter()
    model.fit(x_train, y_train, **fit_kwargs)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_hat = model.predict(x_test)
    t_pred = time.perf_counter() - t0

    acc, _ = accuracyAndError(y_hat, y_test)
    print(f"{name:36s} acc={acc:.4f}  fit={t_fit:6.2f}s  pred={t_pred:5.2f}s")
    return acc


def main():
    X_train, Y_train, X_test = loadFile('data.npz')
    x_train, x_test, y_train, y_test = splitData(
        X_train, Y_train, testSize=TEST_SIZE, randomState=RANDOM_STATE)
    n_train, _ = x_train.shape

    print("\nConfiguração:")
    print(f"  split  = {SPLIT_METHOD}")
    print(f"  gain   = {GAIN_METHOD}   maxDepth = {MAX_DEPTH}   LDA = {LDA_COMPONENTS}")
    print(f"  forest = treeCount={TREE_COUNT} featuresPerTree={FEATURES_PER_TREE} "
          f"samplesFraction={SAMPLES_FRACTION} bootstrap={REPEATED_SAMPLING}")

    if RUN_SINGLE_TREE:
        print("\n" + 25 * "=" + " ÁRVORE ÚNICA " + 25 * "=")
        tree = JojiTree(maxDepth=MAX_DEPTH, gainMethod=GAIN_METHOD, splitMethod=SPLIT_METHOD)
        _evaluate("Tree", tree, x_train, y_train, x_test, y_test,
                  lda_components=LDA_COMPONENTS)

    if RUN_FOREST:
        print("\n" + 25 * "=" + " FLORESTA " + 25 * "=")
        forest = JojiForest(
            featuresPerTree=FEATURES_PER_TREE,
            samplesPerTree=int(n_train * SAMPLES_FRACTION),
            repeatedSampling=REPEATED_SAMPLING,
            treeCount=TREE_COUNT,
            maxDepth=MAX_DEPTH,
            gainMethod=GAIN_METHOD,
            splitMethod=SPLIT_METHOD,
            lda_components=LDA_COMPONENTS,
            n_jobs=N_JOBS,
        )
        _evaluate("Forest", forest, x_train, y_train, x_test, y_test)

    if GENERATE_SUBMISSION:
        print("\n" + 25 * "=" + " SUBMISSÃO KAGGLE " + 25 * "=")
        # Treina no X_train INTEIRO (sem reservar validação) e prevê o X_test real
        n_full = len(X_train)
        submissionForest = JojiForest(
            featuresPerTree=FEATURES_PER_TREE,
            samplesPerTree=int(n_full * SAMPLES_FRACTION),
            repeatedSampling=REPEATED_SAMPLING,
            treeCount=TREE_COUNT,
            maxDepth=MAX_DEPTH,
            gainMethod=GAIN_METHOD,
            splitMethod=SPLIT_METHOD,
            lda_components=LDA_COMPONENTS,
            n_jobs=N_JOBS,
        )
        submissionForest.fit(X_train, Y_train)
        y_submit = submissionForest.predict(X_test)
        path = exportPredictions(y_submit, outDir=SUBMISSION_DIR)
        print(f"CSV de submissão gerado: {path}  ({len(y_submit)} linhas)")


if __name__ == '__main__':
    main()
