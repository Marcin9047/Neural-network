import numpy as np
from numpy import array
from typing import List, Tuple, Callable
from layer_functions import BaseLayerFunction


class LayerBase:
    def __init__(
        self,
        neuron_size,
        wraping_function: BaseLayerFunction,
    ):
        self.neuron_size = neuron_size
        self.activation_function = wraping_function
        self.b = np.random.normal(loc=0, scale=1, size=self.neuron_size)
        # self.b = np.zeros(shape=self.neuron_size)

        self.b_size = self.neuron_size
        self.last_activation = None
        self.last_output = None

    def set_previous_size(self, size):
        self.previous_layer_size = size
        self.w_size = (self.previous_layer_size, self.neuron_size)
        # self.w_size = (self.neuron_size, self.previous_layer_size)

        self.w = np.random.normal(loc=0, scale=1, size=self.w_size)
        self.full_size_of_w = self.neuron_size * self.previous_layer_size

    def compute_deriv_cost_after_w(self, deriv_cost_after_next_layer: array) -> array:
        # func_prim = self.activation_function.get_function_derivative(self.last_z)
        # az_prim = np.dot(self.last_activation.T, func_prim)
        # cost_after_w = np.dot(az_prim, deriv_cost_after_next_layer)
        func_prim = self.activation_function.get_function_derivative(self.last_z)
        # az_prim = np.dot(self.last_activation, func_prim)
        # cost_after_w = np.dot(az_prim, deriv_cost_after_next_layer)
        v1 = np.multiply(func_prim, deriv_cost_after_next_layer)
        v2 = np.dot(np.transpose(self.last_activation), v1)
        return v2

    def compute_deriv_cost_after_b(self, deriv_cost_after_next_layer: array) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)
        cost_after_b = np.multiply(func_prim, deriv_cost_after_next_layer)
        return cost_after_b

    def compute_deriv_cost_after_this_layer(
        self, deriv_cost_after_next_layer: array
    ) -> array:
        func_prim = self.activation_function.get_function_derivative(self.last_z)
        # wz_prim = np.dot(self.w, func_prim.T)
        # cost_after_layer = np.dot(deriv_cost_after_next_layer.T, wz_prim)
        v1 = np.multiply(func_prim, deriv_cost_after_next_layer)
        v2 = np.dot(v1, self.w.T)
        return v2

    def compute_output(self, a0: array):
        # oblicza output warstwy na podstawie wejścia z outputu poprzedniej warstwy
        self.last_activation = a0
        z = np.dot(a0, self.w)
        z += self.b
        self.last_z = z
        a_out = self.activation_function.get_output(z)
        self.last_output = a_out
        return a_out

    def update_w(self, new_w: array):
        self.w = new_w

    def update_b(self, new_bias: array):
        #  Updateuje bias na podstawie płaskiego wektora biasów
        # if len(self.b) != len(new_bias):
        #     raise ValueError(
        #         "there are different number of bias_n than in the old bias"
        #     )
        self.b = new_bias
