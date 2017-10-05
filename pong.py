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
        #self.hidden_layers.append(np.empty(hidden_sizes[0]))
        for nr in range(1, len(hidden_sizes)):
            #self.hidden_layers.append(np.empty(hidden_sizes[nr]))
            w = init_weight(hidden_sizes[nr - 1], hidden_sizes[nr])
            self.weights.append(w)
        #self.hidden_layers.append(np.empty(hidden_sizes[-1]))
        w_out = init_weight(hidden_sizes[-1], out_size)
        self.weights.append(w_out)
        self.out_size = out_size
        self.eta = eta

    def forward_pass(self, x):
        h_in = x.dot(self.weights[0])
        h_in = activation_function(h_in)
        hs = [h_in]
        for nr in range(1, len(self.weights) - 1):
            h = self.weights[nr].dot(hs[nr - 1])
            h = activation_function(h)
            hs.append(h)
        self.hidden_layers.append(hs)
        scores = hs[-1].dot(self.weights[-1])
        return scores

    def backward_pass(self, nr, x, dscores):
        dw_out = np.outer(dscores, self.hidden_layers[nr][-1].T)
        dws = [dw_out]
        dhidden = np.dot(dscores, self.weights[-1].T)
        dhiddens = [dhidden]
        if len(self.hidden_layers[nr]) > 1:
            for i in range(len(self.hidden_layers[nr]) - 1, -1, -1):
                dhidden = np.dot(dhiddens[-1], self.weights[i].T)
                dhidden = activation_function_gradient(dhidden, self.hidden_layers[nr][i])
                if i == 0:
                    dw = x.T.dot(dhidden)
                else:
                    dw = self.hidden_layers[nr][i - 1].T.dot(dhidden)
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


def train(env):
    env.reset()
    input_size = 80 * 80
    hidden_sizes = [200, 200]
    out_size = 3
    eta = 1e-4
    gamma = 0.99
    render = False
    rolluts = 20
    actions = [0]
    observations = []
    current_gradients = []
    net = PolicyNetwork(input_size, hidden_sizes, out_size, eta)
    while rolluts > 0:
        action = actions[-1] + 1
        observation, reward, done, info = env.step(action)
        if render:
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
                net.backward_pass(i, observations[i], current_gradients[i])
            net.update_parameters()
            current_gradients = []
            observations = []
            rolluts -= 1
            env.reset()


env = gym.make("Pong-v0")
train(env)
