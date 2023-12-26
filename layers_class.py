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
        self.b_size = self.neuron_size
        self.last_activation = None
        self.last_output = None

    def set_previous_size(self, size):
        self.previous_layer_size = size
        self.w_size = (self.previous_layer_size, self.neuron_size)
        self.w = np.random.normal(loc=0, scale=1, size=self.w_size)
        self.full_size_of_w = self.neuron_size * self.previous_layer_size

    def compute_deriv_w(self, arguments_todo) -> array:
        pass

    def compute_deriv_w_after_s(self, next_layer_deriv_w_after_s: array) -> array:
        # @TODO Na razie oblicza wpływ poszczególnych wag na koszt w ostatniej warstwie,
        # nie uwzględnia funkcji obliczającej tylko wstępnie oblicza ten wpływ.
        wector_of_activ = np.zeros(self.neuron_size)
        for i, activation_deriv_layer_per_n in enumerate(next_layer_deriv_w_after_s):
            wector_of_activ[i] = np.sum(activation_deriv_layer_per_n)

        new_deriv = np.zeros(self.w_size)
        for i_n in range(self.neuron_size):
            for i_a in range(self.previous_layer_size):
                new_deriv[i_a][i_n] = self.w[i_a][i_n] * wector_of_activ[i_n]
        return new_deriv

    def output(self, a0: array):
        # oblicza output warstwy na podstawie wejścia z outputu poprzedniej warstwy
        self.last_activation = a0
        value = self.activation_function.get_output(self.w, a0, self.b)
        self.last_output = value
        return value

    def update_w(self, new_w: array):
        self.w = new_w

    def update_b(self, new_bias: array):
        #  Updateuje bias na podstawie płaskiego wektora biasów
        if len(self.b) != len(new_bias):
            raise ValueError(
                "there are different number of bias_n than in the old bias"
            )
        self.b = new_bias
