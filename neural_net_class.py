from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List


class CostFunction:
    def __init__(self):
        pass

    def get_cost(self, y_pref, a_output):
        return (a_output - y_pref) ** 2

    def get_float_cost(self, y_pref, a_output):
        return np.sum(self.get_cost(y_pref, a_output))

    def get_deriv_cost_to_a_output(self, y_pref, a_output):
        return 2 * (a_output - y_pref)


class BaseNeuralNetwork:
    def __init__(self, list_of_layers: List[LayerBase]):
        self.layers = list_of_layers
        self.layer_number = len(list_of_layers)
        self.input_size = list_of_layers[0].activation_size
        self.activations_hist = []
        self.flat_w_size = len(self.get_flattened_weights())
        self.flat_b_size = len(self.get_flattened_bias())

    def calculate_output(self, x: array):
        # Przepepuszcza aktywacje przez wszystkie warstwy i zwraca aktywację ostatniej warstwy
        a = x
        activations = []
        activations.append(a)
        for ix, layer in enumerate(self.layers):
            a = layer.compute_activation(a)
            activations.append(a)
        self.activations_hist.append(activations)
        return a

    def _convert_w_to_list(self, w: array) -> array:
        list_w_val = np.ravel(w)

        return list_w_val

    def _convert_list_to_w(self, list_w: array, w_size: Tuple[int, int]) -> array:
        # zmienia vector wartości wag na macierz wag
        w = list_w.reshape(w_size)
        return w

    def get_flattened_weights(self):
        # zwraca vector wszystkich płaskich wag dla wszystkich warstw
        full_weights = []
        for layer in self.layers:
            full_weights.append(self._convert_w_to_list(layer.w))
        return np.concatenate(full_weights)

    def get_flattened_bias(self):
        # zwraca vector wszystkich płaskich biasów dla wszystkich warstw
        full_bias = []
        for layer in self.layers:
            full_bias.append(layer.b)
        return np.concatenate(full_bias)

    def get_flattened_ws_bs(self):
        # Zwraca wektor wszystkich wag i biasów dla wszystkich warstw (wykorzystywany do optymalizacji parametrów całego neural network)
        ws = self.get_flattened_weights()
        bs = self.get_flattened_bias()
        all_s = []
        for w in ws:
            all_s.append(w)
        for b in bs:
            all_s.append(b)
        return array(all_s)

    def update_with_flattened_bias(self, flattened_bias: array):
        # zmienia biasy warstw na te podane w wektorze biasów wszystkich warstw
        flat_sizes = []
        for layer in self.layers:
            flat_sizes.append(layer.b_size)
        vector_list = np.split(flattened_bias, np.cumsum(flat_sizes)[:-1])
        for i in range(len(self.layers)):
            self.layers[i].update_b_with_flat(vector_list[i])

    def update_with_flattened_w_and_b(self, flattened_w_b: array):
        # zmienia biasy i wagi warstw na te podane w wektorze biasów i wag wszystkich warstw
        flat_ws = flattened_w_b[: self.flat_w_size]
        flat_bs = flattened_w_b[self.flat_w_size :]
        self.update_with_flattened_bias(flat_bs)
        self.update_with_flattened_weights(flat_ws)

    def update_with_flattened_weights(self, flattened_ws: array):
        # zmienia wagi warstw na te podane w wektorze wag wszystkich warstw
        flat_sizes = []
        for layer in self.layers:
            flat_sizes.append(layer.full_size_of_w)
        vector_list = np.split(flattened_ws, np.cumsum(flat_sizes)[:-1])
        for i in range(len(self.layers)):
            self.layers[i].update_w_with_flat(vector_list[i])
