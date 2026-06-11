import numpy as np
from sklearn.model_selection import train_test_split

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
    shuffle: bool = False
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=testSize,
        random_state=randomState,
        shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test
