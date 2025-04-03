
import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, epochs=1000):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.randn(n_visible, n_hidden) * 0.1
        self.h_bias = np.zeros(n_hidden)
        self.v_bias = np.zeros(n_visible)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, data):
        for epoch in range(self.epochs):
            for sample in data:
                pos_hidden_activations = np.dot(sample, self.weights) + self.h_bias
                pos_hidden_probs = self.sigmoid(pos_hidden_activations)
                pos_hidden_states = pos_hidden_probs > np.random.rand(self.n_hidden)
                pos_associations = np.outer(sample, pos_hidden_probs)

                neg_visible_activations = np.dot(pos_hidden_states, self.weights.T) + self.v_bias
                neg_visible_probs = self.sigmoid(neg_visible_activations)
                neg_hidden_activations = np.dot(neg_visible_probs, self.weights) + self.h_bias
                neg_hidden_probs = self.sigmoid(neg_hidden_activations)
                neg_associations = np.outer(neg_visible_probs, neg_hidden_probs)

                self.weights += self.learning_rate * (pos_associations - neg_associations)
                self.v_bias += self.learning_rate * (sample - neg_visible_probs)
                self.h_bias += self.learning_rate * (pos_hidden_probs - neg_hidden_probs)

    def run_visible(self, data):
        hidden_activations = np.dot(data, self.weights) + self.h_bias
        hidden_probs = self.sigmoid(hidden_activations)
        return hidden_probs

    def run_hidden(self, data):
        visible_activations = np.dot(data, self.weights.T) + self.v_bias
        visible_probs = self.sigmoid(visible_activations)
        return visible_probs

data = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
rbm = RBM(n_visible=4, n_hidden=2, learning_rate=0.1, epochs=1000)
rbm.train(data)



import networkx as nx
import matplotlib.pyplot as plt

def visualize_rbm(rbm):
    G = nx.Graph()

    for i in range(rbm.n_visible):
        G.add_node(f'v{i}', layer='visible')
    for j in range(rbm.n_hidden):
        G.add_node(f'h{j}', layer='hidden')

    for i in range(rbm.n_visible):
        for j in range(rbm.n_hidden):
            G.add_edge(f'v{i}', f'h{j}')

    pos = {}
    pos.update((f'v{i}', (0, i)) for i in range(rbm.n_visible))
    pos.update((f'h{j}', (1, j)) for j in range(rbm.n_hidden))

    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_color='black')
    plt.title('RBM Architecture')
    plt.show()

visualize_rbm(rbm)


