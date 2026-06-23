import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime

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
    y_test: np.ndarray
):
    dataframe = pd.DataFrame({
        'ID': np.arange(1, len(y_test) + 1),
        'Prediction': y_test
    })

    now = datetime.now().isoformat()

    print(now)

    dataframe.to_csv(f"outputs/predictions_{now}.csv", index=False)

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
