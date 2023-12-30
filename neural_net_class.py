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
        return y_pref - a_output

    def average_lost(self, y_pref_arr, a_output_arr):
        cost = 0
        for i in range(len(y_pref_arr)):
            y_pref = y_pref_arr[i]
            a_output = a_output_arr[i]
            cost += self.get_cost(self, y_pref, a_output)
        return cost / len(y_pref_arr)

    def average_deriv(self, y_pref_arr, a_output_arr):
        deriv = 0
        for i in range(len(y_pref_arr)):
            y_pref = y_pref_arr[i]
            a_output = a_output_arr[i]
            deriv += self.get_deriv_cost_to_a_output(y_pref, a_output)
        return deriv / len(y_pref_arr)


class Neural_net:
    def __init__(self, list_of_layers: List[LayerBase], input_size):
        self.layers = list_of_layers
        self.layer_number = len(list_of_layers)
        self.activations_hist = []

        self.layers[0].set_previous_size(input_size)

        for ind in range(1, len(self.layers)):
            self.layers[ind].set_previous_size(self.layers[ind - 1].neuron_size)

        self.input_size = list_of_layers[0].previous_layer_size

    def calculate_output(self, xIn: array):
        # Przepepuszcza aktywacje przez wszystkie warstwy i zwraca aktywację ostatniej warstwy
        a = xIn
        activations = []
        activations.append(a)
        for ix, layer in enumerate(self.layers):
            a = layer.compute_output(a)
            activations.append(a)
        self.activations_hist.append(activations)
        return a

    def calculate_multiple_output(self, xIn):
        a = xIn
        activations = []
        activations.append(a)
        for ix, layer in enumerate(self.layers):
            a = layer.compute_multiple_output(a)
            activations.append(a)
        self.activations_hist.append(activations)
        return a

    def backpropagation(self, yexp):
        grad_lists = []
        grad_lists.append(self.output_grad(yexp))
        num = len(self.layers) - 2
        for ind in reversed(range(num + 1)):
            grad_lists.append(self.inner_grad(ind))
        return grad_lists  # Lista gradientów 1 wymiar = warstwa od końca 2 wymiar = neurony po kolei

    def inner_grad(self, layerInd):
        output = self.layers[layerInd].last_output
        output_back = self.layers[layerInd - 1].last_output
        inner_grad = np.zeros(len(output))
        new_delta = 0
        for ind, one in enumerate(output):
            output_back1 = output_back[ind][0]
            ones_vector = np.ones(len(one.T))
            act1 = np.dot(np.subtract(ones_vector, one.T), one.T)
            delta_part = np.dot(self.layers[layerInd + 1].w, self.last_delta)
            if len(delta_part[0]) != len(act1.T):
                delta1 = np.dot(act1.T, delta_part)
            else:
                delta1 = np.dot(act1, delta_part.T)
            new_delta += delta1
            out_matrix = []
            for i in range(len(one.T)):
                out_matrix.append(output_back1.T)
            out_matrix = np.array(out_matrix)
            inner_grad1 = np.dot(delta1, out_matrix)
            inner_grad = [a + b for a, b in zip(inner_grad, inner_grad1)]
        self.last_delta = new_delta / len(output)
        return [
            a / len(output) for a in inner_grad[0]
        ]  # Gradient ostatni, zmienia ostatnie wagi

    def output_grad(
        self, yexp
    ):  # aIn - wejściowy x do warstwy yex = wartość oczekiwana
        # input_layer = self.layers[-2]
        output = self.layers[-1].last_output
        output_back = self.layers[-2].last_output
        output_grad = np.zeros(len(output))
        self.last_delta = 0
        for ind, one in enumerate(output):
            output_back1 = output_back[ind]
            cost = CostFunction()
            lose_deriv = cost.get_deriv_cost_to_a_output(yexp[ind], one)
            ones_vector = np.ones(len(one))
            act_deriv = np.dot(
                np.subtract(ones_vector, one), one
            )  # pochodna funkcji aktywacji
            delta = np.dot(lose_deriv, act_deriv)
            self.last_delta += delta
            output_grad1 = np.dot(delta, output_back1)
            output_grad = [a + b for a, b in zip(output_grad, output_grad1)]
        self.last_delta = self.last_delta / len(output)
        return [
            a / len(output) for a in output_grad[0]
        ]  # Gradient ostatni, zmienia ostatnie wagi

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
