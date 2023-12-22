import numpy as np
from numpy import array
from typing import List, Tuple, Callable
from scipy.optimize import minimize


class Layer_function:
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


class SigmoidFunction(Layer_function):
    def get_output(self, weights: array, input_activation: array, bias: array):
        # print(f"w:\n{weights}\nactivation:\n{input_activation}\nbias\n{bias}")
        value = np.dot(input_activation, weights)
        # print(f"value:\n{value}")
        value += bias
        # print(f"value + bias:\n{value}")
        # @TODO SHIT
        val_exp = 1 / (1 + np.exp(-value))
        return np.array(val_exp)


class LayerBase:
    def __init__(
        self,
        neuron_size,
        activation_size,
        wraping_function: Layer_function,
        log_changes=False,
    ):
        self.w_size = (activation_size, neuron_size)
        self.neuron_size = neuron_size
        self.activation_size = activation_size
        self.activation_function = wraping_function
        self.w = np.random.normal(loc=0, scale=1, size=self.w_size)
        self.log_changes = log_changes
        if self.log_changes:
            self.w_history = [self.w]
        self.b = np.random.normal(loc=0, scale=1, size=self.neuron_size)
        if self.log_changes:
            self.b_history = [self.b]
        self.b_size = self.neuron_size
        self.last_activation = None
        self.last_output = None
        self.full_size_of_w = self.neuron_size * self.activation_size

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
        # if len(a0) != self.activation_size:
        #     raise ValueError("Layer input is different size than activation size")
        self.last_activation = a0
        value = self.activation_function.get_output(self.w, a0, self.b)
        self.last_output = value
        return value

    def update_w(self, new_w: array):
        if self.log_changes:
            self.w_history.append(new_w)
        # if new_w.size != self.w_size:
        #     raise ValueError("new weights are a different shape than old weights")
        self.w = new_w

    def update_w_with_flat(self, new_flat_w: array):
        if len(new_flat_w) != self.full_size_of_w:
            raise ValueError(
                "there are different number of weights than in the old wector"
            )
        new_w = new_flat_w.reshape(self.w_size)
        self.update_w(new_w)

    def update_b_with_flat(self, new_flat_bias: array):
        if len(self.b) != len(new_flat_bias):
            raise ValueError(
                "there are different number of bias_n than in the old bias"
            )
        if self.log_changes:
            self.b_history.append(new_flat_bias)
        self.b = new_flat_bias


class Cost_function:
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
        a = x
        activations = []
        activations.append(a)
        for ix, layer in enumerate(self.layers):
            a = layer.compute_activation(a)
            activations.append(a)
        self.activations_hist.append(activations)
        return a

    def _convert_w_to_list(self, w: array) -> array:
        # list_w_val = []
        # for row in w:
        #     for val in row:
        #         list_w_val.append(val)
        list_w_val = np.ravel(w)

        return list_w_val

    def _convert_list_to_w(self, list_w: array, w_size: Tuple[int, int]) -> array:
        # self.w_size = (activation_size, neuron_size)
        w = list_w.reshape(w_size)
        return w

    def get_flattened_weights(self):
        full_weights = []
        for layer in self.layers:
            full_weights.append(self._convert_w_to_list(layer.w))
        return np.concatenate(full_weights)

    def get_flattened_bias(self):
        full_bias = []
        for layer in self.layers:
            full_bias.append(layer.b)
        return np.concatenate(full_bias)

    def get_flattened_ws_bs(self):
        ws = self.get_flattened_weights()
        bs = self.get_flattened_bias()
        # print(ws)
        # print(bs)
        all_s = []
        for w in ws:
            all_s.append(w)
        for b in bs:
            all_s.append(b)
        return array(all_s)

    def update_with_flattened_bias(self, flattened_bias: array):
        flat_sizes = []
        for layer in self.layers:
            flat_sizes.append(layer.b_size)
        vector_list = np.split(flattened_bias, np.cumsum(flat_sizes)[:-1])
        for i in range(len(self.layers)):
            self.layers[i].update_b_with_flat(vector_list[i])

    def update_with_flattened_w_and_b(self, flattened_w_b: array):
        flat_ws = flattened_w_b[: self.flat_w_size]
        flat_bs = flattened_w_b[self.flat_w_size :]
        self.update_with_flattened_bias(flat_bs)
        self.update_with_flattened_weights(flat_ws)

    # def _convert_list_of_flat_ws_to_ws(self, flattened_ws: array):
    #     flat_sizes = []
    #     tuple_sizes = []
    #     for layer in self.layers:
    #         flat_sizes.append(layer.full_size_of_w)
    #         tuple_sizes.append(layer.w_size)
    #     vector_list = np.split(flattened_ws, np.cumsum(flat_sizes)[:-1])

    #     rel_ws = []
    #     for i, vector_w in enumerate(vector_list):
    #         rel_ws.append()
    def update_with_flattened_weights(self, flattened_ws: array):
        flat_sizes = []
        for layer in self.layers:
            flat_sizes.append(layer.full_size_of_w)
        vector_list = np.split(flattened_ws, np.cumsum(flat_sizes)[:-1])
        for i in range(len(self.layers)):
            self.layers[i].update_w_with_flat(vector_list[i])


# class SolverRequirements:
#     def __init__(self):
#         self.weights_grad = False
#         self.biases_grad = False
#         self.layers = False
#         self.previous_layer_weights = False
#         self.previous_layer_biases = False
#         self.cost_per_layer = False
#         self.abs_cost = False


# class SolverComputeParams:
#     def __init__(self):
#         # List[array]
#         self.layers_weights = None
#         self.layers_biases = None
#         self.weights_grad = None
#         self.biases_grad = None
#         # List[LayerBase]
#         self.layers = None


# class SolverSolves:
#     def __init__(self):
#         self.new_layers_weights = None
#         self.new_layers_biases = None


# class Base_solver_class:
#     def __init__(self, Bw, Bb):
#         self.Bw = Bw
#         self.Bb = Bb
#         self._make_requirement_list()

#     def compute_new_weights(self, scp: SolverComputeParams):
#         new_weights = []
#         for ix, w in enumerate(scp.layers_weights):
#             new_weights.append(w - (self.Bw * scp.weights_grad[ix]))
#         return new_weights

#     def compute_new_bias(self, scp: SolverComputeParams):
#         new_biases = []
#         for ix, w in enumerate(scp.layers_biases):
#             new_biases.append(w - (self.Bb * scp.biases_grad[ix]))
#         return new_biases

#     def compute_solves(self, scp: SolverComputeParams):
#         ss = SolverSolves()
#         ss.new_layers_biases = self.compute_new_bias(scp)
#         ss.new_layers_weights = self.compute_new_weights(scp)
#         return ss

#     def _check_for_requirements(self, scp: SolverComputeParams):
#         pass

#     def get_requirements(self):
#         return self.s_req

#     def _make_requirement_list(self):
#         sr = SolverRequirements()
#         sr.biases_grad = True
#         sr.weights_grad = True
#         sr.previous_layer_biases = True
#         sr.previous_layer_weights = True
#         self.s_req = sr


def testing_func(x):
    val = (x**2) * np.sin(x)
    val += 100 * np.sin(x) * np.cos(x)
    return val


def optimise_with_scipy(
    emulate_func, n_m: BaseNeuralNetwork, limits, nr_per_iter: int, max_iter=100000
):
    cf = Cost_function()
    amplitude = limits[1] - limits[0]
    cost_history = []
    import cma

    def cost_func(params: array):
        n_m.update_with_flattened_w_and_b(params)
        # random_list = np.random.rand(nr_per_iter)
        # x_list = (random_list * amplitude) + limits[0]
        x_list = np.linspace(limits[0], limits[1], nr_per_iter)
        y_list = emulate_func(x_list)
        y_pred = []
        cost = 0
        for x in x_list:
            y_pred.append(n_m.calculate_output(x))
        cost += cf.get_float_cost(y_list, np.array(y_pred))
        cost_history.append(cost)
        return cost / nr_per_iter

    # cf.get_cost()
    # cost_func = cf.get_cost
    params_init = n_m.get_flattened_ws_bs()
    # print(params_init)
    # mini_result = minimize(
    #     cost_func,
    #     params_init,
    # )
    es = cma.CMAEvolutionStrategy(params_init, 0.5, {"maxiter": max_iter})
    es.optimize(cost_func)

    return es.result.xbest, n_m.activations_hist, cost_history


def get_values_for_X(X, nm: BaseNeuralNetwork):
    Y = []
    for x in X:
        Y.append(float(nm.calculate_output(x)))
    return Y


def sin_function(x):
    return np.sin(x)


if __name__ == "__main__":
    f = SigmoidFunction()
    l1 = LayerBase(5, 1, f)
    l2 = LayerBase(10, 5, f)
    l3 = LayerBase(10, 10, f)
    l4 = LayerBase(5, 10, f)
    # l5 = LayerBase(5, 5, f)
    l_out = LayerBase(1, 5, f)

    n_manage = BaseNeuralNetwork([l1, l2, l3, l4, l_out])
    first_ws_bs = n_manage.get_flattened_ws_bs()
    result, activations_hist, cost_his = optimise_with_scipy(
        sin_function, n_manage, (-5, 5), 500, 100
    )

    print(result)
    # print(mini_result["x"])
    # print(activations_hist)
    best_ws_bs = result

    from matplotlib import pyplot as plt

    # plt.plot(cost_his)
    # plt.show()
    n_manage.update_with_flattened_w_and_b(best_ws_bs)
    x_plot = np.linspace(-7, 7, 1000)
    y_perf = sin_function(x_plot)
    y_best = get_values_for_X(x_plot, n_manage)
    n_manage.update_with_flattened_w_and_b(first_ws_bs)
    y_first = get_values_for_X(x_plot, n_manage)
    plt.plot(x_plot, y_perf, label="function")
    # plt.plot(x_plot, y_first, label="random weights")
    plt.plot(x_plot, y_best, label="weigths trained")
    plt.legend()
    # plt.semilogy()
    plt.show()
    # w = l2.b
    # w_flat = n_manage._convert_w_to_list(w)
    # re_w = n_manage._convert_list_to_w(w_flat, l2.w_size)
    # print(f"{w}\n\n{w_flat}\n\n{re_w}")
    # print((w - re_w))
#     a_1 = l1.compute_activation(np.array([1]))
#     a_2 = l2.compute_activation(a_1)
#     a_out = l_out.compute_activation(a_2)

#     x = 1
#     y = testing_func(x)
#     print(x, y)

#     c_f = Cost_function()
#     e_l = c_f.get_cost(y, float(a_out))
#     de_da_l = c_f.get_deriv_cost_to_a_output(y, float(a_out))
#     deriv_w_l_out = l_out.compute_deriv_w_after_s(array([de_da_l]))
#     deriv_w_l_2 = l2.compute_deriv_w_after_s(deriv_w_l_out)
#     deriv_w_l_1 = l1.compute_deriv_w_after_s(deriv_w_l_2)

#     print("l1", a_1, "\nw1:\n", l1.w, "\ndw1:\n", deriv_w_l_1)
#     print("\n")
#     print("l2", a_2, "\nw2:\n", l2.w, "\ndw2:\n", deriv_w_l_2)
#     print("\n")
#     print("lout", a_out, "\nw out:\n", l_out.w, "\nd out:\n", deriv_w_l_out)
#     print("\n")
#     print("deriv_of_y")
#     print(e_l, de_da_l)
