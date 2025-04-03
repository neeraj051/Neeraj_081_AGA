import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Dummy dataset (binary stack representation)
data = np.array([
    [1, 0, 0, 0],  # Stack with one item
    [1, 1, 0, 0],  # Stack with two items
    [1, 1, 1, 0],  # Stack with three items
    [1, 1, 1, 1],  # Full stack
    [0, 0, 0, 0],  # Empty stack
])

class StackedRBM(BaseEstimator, TransformerMixin):
    def __init__(self, layer_sizes=[4, 3, 2], learning_rate=0.1, n_iter=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.rbms = []

    def fit(self, X):
        layer_input = X
        for i in range(len(self.layer_sizes) - 1):
            rbm = BernoulliRBM(n_components=self.layer_sizes[i+1],
                               learning_rate=self.learning_rate,
                               n_iter=self.n_iter,
                               verbose=True)
            rbm.fit(layer_input)
            self.rbms.append(rbm)
            layer_input = rbm.transform(layer_input)
        return self

    def transform(self, X):
        layer_input = X
        for rbm in self.rbms:
            layer_input = rbm.transform(layer_input)
        return layer_input

# Define and train stacked RBM
stacked_rbm = StackedRBM(layer_sizes=[4, 3, 2], learning_rate=0.1, n_iter=2000)
stacked_rbm.fit(data)

# Transform input data through the stacked RBM
encoded_data = stacked_rbm.transform(data)
print("\nEncoded representation:\n", encoded_data)
