from math import sqrt
import numpy as np
from random import randint
import gym

def init_weight(in_size, out_size):
    return np.random.randn(in_size, out_size) * (2 / sqrt(in_size))

class PolicyNetwork:
    hidden_layers = []
    weights = []
    def __init__(self, input_size, hidden_sizes, out_size):
        for size in hidden_sizes:
            self.hidden_layers.append(np.empty(size))

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

def activation_function(x):
    x[x < 0] = 0
    return x

def train(env):
    env.reset()
    env.render()
    done = False
    while not done:
        observation, reward, done, info = env.step(randint(2, 3))
        env.render()
        print(observation)
        print(reward)

env = gym.make("Pong-v0")
train(env)