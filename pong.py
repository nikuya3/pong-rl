from math import sqrt
import numpy as np
from PIL import Image
from random import randint
import gym


def activation_function(x):
    x[x < 0] = 0
    return x


def init_weight(in_size, out_size):
    return np.random.randn(in_size, out_size) * sqrt(2 / in_size)


class PolicyNetwork:
    hidden_layers = []
    weights = []

    def __init__(self, input_size, hidden_sizes, out_size):
        w_in = init_weight(input_size, hidden_sizes[0])
        self.weights.append(w_in)
        for nr in range(1, len(hidden_sizes) - 1):
            self.hidden_layers.append(np.empty(hidden_sizes[nr]))
            w = init_weight(hidden_sizes[nr - 1], hidden_sizes[nr])
            self.weights.append(w)
        w_out = init_weight(hidden_sizes[-1], out_size)
        self.weights.append(w_out)

    def forward_pass(self, x):
        hiddens = []
        h_in = x.dot(self.weights[0])
        h_in = activation_function(h_in)
        hiddens.append(h_in)
        for nr in range(1, len(self.weights) - 1):
            h = hiddens[nr - 1].dot(self.weights[nr])
            h = activation_function(h)
            hiddens.append(h)
        scores = hiddens[-1].dot(self.weights[-1])
        return scores

    def backward_pass(self):
        pass


def preprocess_image(image):
    image = image[34:194]
    image = image[::2, ::2, 0]
    image[image == 144] = 0
    #image[image != 0] = 0
    image = image.flatten()
    return image


def train(env):
    env.reset()
    #env.render()
    input_size = 80 * 80
    hidden_sizes = [200, 200]
    out_size = 3
    done = False
    count = 0
    net = PolicyNetwork(80 * 80, hidden_sizes, 3)
    while not done:
        count += 1
        observation, reward, done, info = env.step(randint(2, 3))
        #env.render()
        observation = preprocess_image(observation)
        scores = net.forward_pass(observation)
        print(scores)
        if count == 100:
            img = Image.fromarray(observation)
            img.show()
            done = True


env = gym.make("Pong-v0")
train(env)
