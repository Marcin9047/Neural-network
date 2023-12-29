import numpy as np
from numpy import array
from typing import List, Tuple, Callable


class BaseLayerFunction:
    def get_output(self, z: array):
        # print(f"w:\n{weights}\nactivation:\n{input_activation}\nbias\n{bias}

        return z

    def get_function_derivative(self, z: array):
        return z

    # def reverse_output(self, output_activation: array, weights: array, bias: array):
    #     output_activation -= bias
    #     input_activation_reconstructed = np.linalg.solve(weights, output_activation)
    #     return input_activation_reconstructed


class SigmoidFunction(BaseLayerFunction):
    def get_output(self, z: array):
        val_exp = 1 / (1 + np.exp(-z))
        return np.array(val_exp)

    def get_function_derivative(self, z: array):
        val_exp = 1 / (1 + np.exp(-z))
        return np.array(val_exp * (1 - val_exp))

    # def get_derivative_after_w(self, weights: array, bias: array, dC: array):
    #     value =
    #     return dC
