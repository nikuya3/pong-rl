from math import sqrt
import numpy as np
from random import randint
import gym


def activation_function(x):
    x[x < 0] = 0
    return x


def activation_function_gradient(dx, x):
    dx[x < 0] = 0
    return dx


def init_weight(in_size, out_size):
    return np.random.randn(in_size, out_size) * sqrt(2 / in_size)


class PolicyNetwork:
    hidden_layers = []
    weights = []
    weight_grads = []

    def __init__(self, input_size, hidden_sizes, out_size, eta):
        w_in = init_weight(input_size, hidden_sizes[0])
        self.weights.append(w_in)
        for nr in range(1, len(hidden_sizes) - 1):
            self.hidden_layers.append(np.empty(hidden_sizes[nr]))
            w = init_weight(hidden_sizes[nr - 1], hidden_sizes[nr])
            self.weights.append(w)
        w_out = init_weight(hidden_sizes[-1], out_size)
        self.weights.append(w_out)
        self.out_size = out_size
        self.eta = eta

    def forward_pass(self, x):
        h_in = x.dot(self.weights[0])
        h_in = activation_function(h_in)
        self.hidden_layers.append(h_in)
        for nr in range(1, len(self.weights) - 1):
            h = self.hidden_layers[nr - 1].dot(self.weights[nr])
            h = activation_function(h)
            self.hidden_layers.append(h)
        scores = self.hidden_layers[-1].dot(self.weights[-1])
        return scores

    def backward_pass(self, x, dscores):
        x = x.reshape(len(x), 1)
        dscores = dscores.reshape(len(dscores), 1)
        dw_out = dscores.dot(self.hidden_layers[-1].T)
        dws = [dw_out]
        dhiddens = [self.weights[-1].T.dot(dscores)]
        if len(self.hidden_layers) > 1:
            for i in range(len(self.hidden_layers) - 1, -1, -1):
                dhidden = dhiddens[-1].dot(self.weights[i + 1].T)
                dhidden = activation_function_gradient(dhidden, self.hidden_layers[i])
                if i == 0:
                    dw = x.T.dot(dhidden)
                else:
                    dw = self.hidden_layers[i - 1].T.dot(dhidden)
                dhiddens.append(dhidden)
                dws.append(dw)
        self.weight_grads.append(dws)

    def update_parameters(self):
        dws = np.array(self.weight_grads)
        dws = np.average(dws, axis=1)
        for i in range(len(dws)):
            self.weights[i] += self.eta * dws[i]


def preprocess_image(image):
    image = image[34:194]
    image = image[::2, ::2, 0]
    image[image == 144] = 0
    #image[image != 0] = 0
    image = image.flatten()
    return image


def probs(scores):
    """
    Calculates the probabilities out of a neural networks class scores.
    :param scores: The score matrix of form (N x K), where N is the number of observations and K is the number of classes.
    :return: The probabilities of the same form as the input scores.
    """
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def calculate_cross_entropy_loss(s, y, w, lambda_):
    """
    Calculates the loss of a score matrix depending on the ground truth labels.
    This method uses cross entropy loss (from Softmax).
    :param s: The score matrix of form (N x K), where N is the number of observations and K is the number of classes.
    :param y: The ground truth label vector of length N.
    :param w: The weight matrix of the output layer of form (H x K), where H is the size of the previous layer.
    :param lambda_: The regularization loss hyperparameter.
    :return: The cross-entropy loss, where 0 indicates a perfect match between s and y
    and +Inf indicates a perfect mismatch.
    """
    probabilities = probs(s)
    log_probabilities = - np.log(probabilities[range(len(y)), y])
    data_loss = np.sum(log_probabilities) / len(y)
    return data_loss


def cross_entropy_loss_gradient(s, y):
    """
    Calculates the gradient of the hinge loss function by the scores.
    The gradient formula is { ds_j / dL = e^s_j / sum e^j, ds_y_i / dL = e^s_y_i / sum e^j - 1 }.
    :param s: The score parameter of the loss function.
    :param y: The ground truth label parameter of the loss function.
    :return: The gradient as a matrix of the same shape as `s`.
    """
    dscores = probs(s)
    dscores[range(len(y)), y] -= 1
    dscores /= len(y)
    return dscores


def train(env):
    env.reset()
    env.render()
    input_size = 80 * 80
    hidden_sizes = [200, 200]
    out_size = 3
    eta = 1e-4
    gamma = 0.99
    rolluts = 20
    actions = [0]
    observations = []
    current_gradients = []
    net = PolicyNetwork(input_size, hidden_sizes, out_size, eta)
    while rolluts > 0:
        action = actions[-1] + 1
        observation, reward, done, info = env.step(action)
        env.render()
        observation = preprocess_image(observation)
        observations.append(observation)
        scores = net.forward_pass(observation)
        # Bias the policy to take the sampled action -> to take clear actions in general
        y = np.argmax(scores)
        dscores = np.full(len(scores), -1)
        dscores[y] = 1
        current_gradients.append(dscores)
        actions.append(y)
        if done:
            for i in range(len(current_gradients)):
                current_gradients[i] = current_gradients[i] * reward
                net.backward_pass(observations[i], current_gradients[i])
            net.update_parameters()
            current_gradients = []
            observations = []
            rolluts -= 1
            env.reset()


env = gym.make("Pong-v0")
train(env)
