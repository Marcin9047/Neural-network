from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from type_converters import get_flattened_weights, get_normal_w


class CostFunction:
    def __init__(self):
        pass

    def get_cost(self, y_pref, a_output):
        values = y_pref - a_output
        return [x**2 for x in values]

    def get_float_cost(self, y_pref, a_output):
        return np.sum(self.get_cost(y_pref, a_output))

    def get_deriv_cost_to_a_output(self, y_pref, a_output):
        return 2 * (a_output - y_pref)


class Neural_net:
    def __init__(self, list_of_layers: List[LayerBase], input_size):
        self.layers = list_of_layers
        self.layer_number = len(list_of_layers)
        self.activations_hist = []

        self.layers[0].set_previous_size(input_size)

        for ind in range(1, len(self.layers)):
            self.layers[ind].set_previous_size(self.layers[ind - 1].neuron_size)

        self.input_size = list_of_layers[0].previous_layer_size

    def calculate_output(self, x: array):
        # Przepepuszcza aktywacje przez wszystkie warstwy i zwraca aktywację ostatniej warstwy
        a = x
        activations = []
        activations.append(a)
        for ix, layer in enumerate(self.layers):
            a = layer.compute_output(a)
            activations.append(a)
        self.activations_hist.append(activations)
        return a

    def update_with_flattened_bias(self, flattened_bias: array):
        # zmienia biasy warstw na te podane w wektorze biasów wszystkich warstw
        flat_sizes = []
        for layer in self.layers:
            flat_sizes.append(layer.b_size)
        vector_list = np.split(flattened_bias, np.cumsum(flat_sizes)[:-1])
        for i in range(len(self.layers)):
            self.layers[i].update_b(vector_list[i])

    def update_with_flattened_w_and_b(self, flattened_w_b: array):
        flat_w_size = len(get_flattened_weights(self.layers))
        # zmienia biasy i wagi warstw na te podane w wektorze biasów i wag wszystkich warstw
        flat_ws = flattened_w_b[:flat_w_size]
        flat_bs = flattened_w_b[flat_w_size:]
        self.update_with_flattened_bias(flat_bs)
        self.update_with_flattened_weights(flat_ws)

    def update_with_flattened_weights(self, flattened_ws: array):
        # zmienia wagi warstw na te podane w wektorze wag wszystkich warstw
        flat_sizes = []
        for layer in self.layers:
            flat_sizes.append(layer.full_size_of_w)
        vector_list = np.split(flattened_ws, np.cumsum(flat_sizes)[:-1])
        for i in range(len(self.layers)):
            wout = get_normal_w(self.layers[i], vector_list[i])
            self.layers[i].update_w(wout)
