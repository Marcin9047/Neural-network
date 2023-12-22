import numpy as np
from numpy import array
from typing import List, Tuple, Callable


class LayerFunction:
    def get_output(self, weights: array, input_activation: array, bias: array):
        # print(f"w:\n{weights}\nactivation:\n{input_activation}\nbias\n{bias}")
        value = np.dot(input_activation, weights)
        # print(f"value:\n{value}")
        value += bias
        # print(f"value + bias:\n{value}")
        # @TODO SHIT
        return np.array(value)

    def get_deriv_output_to_w_input_a(self, stuff):
        pass

    # def reverse_output(self, output_activation: array, weights: array, bias: array):
    #     output_activation -= bias
    #     input_activation_reconstructed = np.linalg.solve(weights, output_activation)
    #     return input_activation_reconstructed


class SigmoidFunction(LayerFunction):
    def get_output(self, weights: array, input_activation: array, bias: array):
        value = np.dot(input_activation, weights)
        value += bias
        val_exp = 1 / (1 + np.exp(-value))
        return np.array(val_exp)
