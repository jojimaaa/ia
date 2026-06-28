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
