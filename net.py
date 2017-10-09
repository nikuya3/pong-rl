from math import floor, sqrt
import numpy as np
from os import listdir, mkdir
from os.path import isdir, join
from pickle import dump, load
from time import time


def activation_function(x):
    x[x < 0] = 0
    return x


def activation_gradient(dx, x):
    dx[x <= 0] = 0
    return dx


def init_weight(in_size, out_size):
    return np.random.randn(in_size, out_size) * sqrt(2 / in_size)


class PolicyNetwork:
    dump_dir = 'model-dump'
    hidden_layers = []
    weights = []
    weight_grads = []

    def __init__(self, input_size, hidden_sizes, out_size):
        w_in = init_weight(input_size, hidden_sizes[0])
        self.weights.append(w_in)
        for nr in range(1, len(hidden_sizes)):
            w = init_weight(hidden_sizes[nr - 1], hidden_sizes[nr])
            self.weights.append(w)
        w_out = init_weight(hidden_sizes[-1], out_size)
        self.weights.append(w_out)
        self.out_size = out_size
        self.reset_hidden_layers(len(hidden_sizes))

    def reset_hidden_layers(self, hidden_amount):
        self.hidden_layers = []
        for nr in range(hidden_amount):
            self.hidden_layers.append([])

    def forward_pass(self, x):
        h_in = x.dot(self.weights[0])
        h_in = activation_function(h_in)
        self.hidden_layers[0].append(h_in)
        hs = [h_in]
        for nr in range(1, len(self.weights) - 1):
            h = self.weights[nr].dot(hs[nr - 1])
            h = activation_function(h)
            self.hidden_layers[nr].append(h)
            hs.append(h)
        scores = hs[-1].dot(self.weights[-1])
        return scores

    def backward_pass(self, x, dscores):
        hs = np.array(self.hidden_layers)
        dw_out = hs[-1].T.dot(dscores)
        dhiddens = {}
        dw = [np.full(w_i.shape, .0) for w_i in self.weights]
        dw[-1] = dw_out
        for h in range(len(hs) - 1, -1, -1):
            if h == len(hs) - 1:
                dhidden = dscores.dot(self.weights[-1].T)
            else:
                dhidden = dhiddens[h + 1].dot(self.weights[h + 1].T)
            dhidden = activation_gradient(dhidden, hs[h])
            dhiddens[h] = dhidden
            if h == 0:
                dw[h] = x.T.dot(dhidden)
            else:
                dw[h] = hs[h - 1].T.dot(dhidden)
        self.weight_grads = dw
        self.reset_hidden_layers(len(hs))

    def update_parameters(self, eta):
        for i in range(len(self.weight_grads)):
            self.weights[i] += eta * self.weight_grads[i]

    def save(self, path=None):
        """
        Saves the current paramters to disk.
        :param path: Specifies the path to the file in which the parameters should be written. If not specified, a new
                    file will be created in `self.dump_dir`.
        """
        if path is None:
            if not isdir(self.dump_dir):
                mkdir(self.dump_dir)
            filename = 'model_' + str(floor(time())) + '.p'
            path = join(self.dump_dir, filename)
        with open(path, 'wb') as file:
            dump(self.weights, file)

    def load(self, path=None):
        """
        Loads saved parameters from storage.
        :param path: Specifies the path to the file in which the parameters were saved. If not specified, the latest
                    file in `self.dump_dir` is chosen.
        """
        if path is None:
            if not isdir(self.dump_dir):
                raise AttributeError('No parameters available')
            filenames = sorted(listdir(self.dump_dir))
            path = join(self.dump_dir, filenames[-1])
        with open(path, 'rb') as file:
            self.weights = load(file)

