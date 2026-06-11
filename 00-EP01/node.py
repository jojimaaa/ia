class Node():
    def __init__(self,
                 w_star=None,
                 th_star=None,
                 left=None,
                 right=None,
                 label=None):

        # for decision node
        self.w_star = w_star
        self.th_star = th_star
        self.left = left
        self.right = right

        # y_hat for leaf nodes
        self.label = label
