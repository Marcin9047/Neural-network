import numpy as np
from numpy import array
from typing import List, Tuple, Callable


class Layer_function:
    def __init__(self):
        pass

    def get_output(self, weights: array, input_activation: array, bias: array):
        value = np.dot(weights.T, input_activation) + bias
        return value

    def get_deriv_output_to_w_input_a(self, stuff):
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
        pass

    def compute_deriv_w_after_s(self, next_layer_deriv_w_after_s: array) -> array:
        # @TODO add function deriv to full derivative
        wector_of_activ = np.zeros(self.neuron_size)
        for i, activation_deriv_layer_per_n in enumerate(next_layer_deriv_w_after_s):
            wector_of_activ[i] = np.sum(activation_deriv_layer_per_n)

        new_deriv = np.zeros(self.w_size)
        for i_n in range(self.neuron_size):
            for i_a in range(self.activation_size):
                new_deriv[i_a][i_n] = self.w[i_a][i_n] * wector_of_activ[i_n]
        return new_deriv

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
        return 2 * (a_output - y_pref)


class SolverRequirements:
    def __init__(self):
        self.weights_grad = False
        self.biases_grad = False
        self.layers = False
        self.previous_layer_weights = False
        self.previous_layer_biases = False
        self.cost_per_layer = False
        self.abs_cost = False


class SolverComputeParams:
    def __init__(self):
        # List[array]
        self.layers_weights = None
        self.layers_biases = None
        self.weights_grad = None
        self.biases_grad = None
        # List[LayerBase]
        self.layers = None


class SolverSolves:
    def __init__(self):
        self.new_layers_weights = None
        self.new_layers_biases = None


class Base_solver_class:
    def __init__(self, Bw, Bb):
        self.Bw = Bw
        self.Bb = Bb
        self._make_requirement_list()

    def compute_new_weights(self, scp: SolverComputeParams):
        new_weights = []
        for ix, w in enumerate(scp.layers_weights):
            new_weights.append(w - (self.Bw * scp.weights_grad[ix]))
        return new_weights

    def compute_new_bias(self, scp: SolverComputeParams):
        new_biases = []
        for ix, w in enumerate(scp.layers_biases):
            new_biases.append(w - (self.Bb * scp.biases_grad[ix]))
        return new_biases

    def compute_solves(self, scp: SolverComputeParams):
        ss = SolverSolves()
        ss.new_layers_biases = self.compute_new_bias(scp)
        ss.new_layers_weights = self.compute_new_weights(scp)
        return ss

    def _check_for_requirements(self, scp: SolverComputeParams):
        pass

    def get_requirements(self):
        return self.s_req

    def _make_requirement_list(self):
        sr = SolverRequirements()
        sr.biases_grad = True
        sr.weights_grad = True
        sr.previous_layer_biases = True
        sr.previous_layer_weights = True
        self.s_req = sr


def testing_func(x):
    val = (x**2) * np.sin(x)
    val += 100 * np.sin(x) * np.cos(x)
    return val


if __name__ == "__main__":
    f = Layer_function()
    l1 = LayerBase(3, 1, f)
    l2 = LayerBase(5, 3, f)
    l_out = LayerBase(1, 5, f)
    a_1 = l1.compute_activation(np.array([1]))
    a_2 = l2.compute_activation(a_1)
    a_out = l_out.compute_activation(a_2)

    x = 1
    y = testing_func(x)
    print(x, y)

    c_f = Cost_function()
    e_l = c_f.get_cost(y, float(a_out))
    de_da_l = c_f.get_deriv_cost_to_a_output(y, float(a_out))
    deriv_w_l_out = l_out.compute_deriv_w_after_s(array([de_da_l]))
    deriv_w_l_2 = l2.compute_deriv_w_after_s(deriv_w_l_out)
    deriv_w_l_1 = l1.compute_deriv_w_after_s(deriv_w_l_2)

    print("l1", a_1, "\nw1:\n", l1.w, "\ndw1:\n", deriv_w_l_1)
    print("\n")
    print("l2", a_2, "\nw2:\n", l2.w, "\ndw2:\n", deriv_w_l_2)
    print("\n")
    print("lout", a_out, "\nw out:\n", l_out.w, "\nd out:\n", deriv_w_l_out)
    print("\n")
    print("deriv_of_y")
    print(e_l, de_da_l)
