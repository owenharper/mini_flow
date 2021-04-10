import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))

class mini_NN:
    def __init__(self, size):
        self.layer=len(size)
        weights=[np.random]