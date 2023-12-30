import math
import numpy as np


class Activation_function:
    def __init__(self, activation_type: str):
        self.type = activation_type

    def get_output(self, weights, input_activation, bias):
        value = np.dot(input_activation, weights)
        value += bias
        if self.type == "logistic":
            return 1 / (1 + np.exp(-value))
        elif self.type == "ReLU":
            return np.where(value < 0, 0, value)
        elif self.type == "TanH":
            return 2 / (1 + np.exp(-2 * value)) - 1

    def get_deriv(self, weights, input_activation, bias):
        value = np.dot(input_activation, weights)
        value += bias
        if self.type == "logistic":
            fout = 1 / (1 + np.exp(-value))
            return fout * (1 - fout)
        elif self.type == "ReLU":
            return np.where(value < 0, 0, 1)
        elif self.type == "TanH":
            fout = 2 / (1 + np.exp(-2 * value)) - 1
            return 1 - (fout**2)
