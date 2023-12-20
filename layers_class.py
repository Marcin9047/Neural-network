import numpy as np
from numpy import array
from typing import List, Tuple, Callable


class Layer_function:
    def __init__(self):
        pass

    def get_output(self, weights: array, input_activation: array, bias: array):
        value = np.dot(weights.T, input_activation) + bias
        return value

    def get_deriv_output_to_input_a(self, stuff):
        pass


class LayerBase:
    def __init__(self, neuron_size, activation_size, wraping_function: Layer_function):
        self.w_size = (activation_size, neuron_size)
        self.neuron_size = neuron_size
        self.activation_size = activation_size
        self.activation_function = wraping_function
        self.w = np.random.normal(loc=0, scale=1, size=self.w_size)
        self.b = np.random.normal(loc=0, scale=1, size=self.neuron_size)
        self.last_activation = None

    def compute_deriv_w(self, arguments_todo) -> array:
        return None

    def compute_activation(self, a0: array):
        value = self.activation_function.get_output(self.w, a0, self.b)
        self.last_activation = value
        return value


class Cost_function:
    def __init__(self):
        pass

    def get_cost(self, y_pref, a_output):
        return (a_output - y_pref) ** 2

    def get_deriv_cost_to_a_output(self, y_pref, a_output):
        return 2(a_output - y_pref)


class Gradient_descent:
    def __init__(self, Bw, Bb):
        self.Bw = Bw
        self.Bb = Bb

    def compute_new_weights(self, weights: List[array], weigths_deriv: List[array]):
        new_weights = []
