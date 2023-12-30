from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import Neural_net, CostFunction
from layer_functions import SigmoidFunction, BaseLayerFunction
from matplotlib import pyplot as plt
import cma
from type_converters import *
from activation_class import Activation_function


class Multiple_evaluation:
    def __init__(
        self,
        emulate_func,
        net: Neural_net,
        limits,
        nr_per_iter: int,
    ):
        self.net = net
        self.function = emulate_func
        self.limits = limits
        self.nr_per_iter = nr_per_iter

    def cost_func(self, params: array):
        cf = CostFunction()
        self.net.update_with_flattened_w_and_b(params)
        x_list = np.linspace(self.limits[0], self.limits[1], self.nr_per_iter)
        y_list = self.function(x_list)
        self.x_test = x_list
        self.exp = y_list
        y_pred = []
        for x in x_list:
            y_pred.append(self.net.calculate_output(x)[0][0])
        cost = cf.get_cost(y_list, y_pred)
        val = np.sum(cost)

        return val

    def optimise_with_evolutions(
        self,
        max_iter=100000,
        popsize=1000,
        sigma0=0.1,
    ):
        params_init = get_flattened_ws_bs(self.net.layers)

        es = cma.CMAEvolutionStrategy(
            params_init, sigma0, {"popsize": popsize, "maxiter": max_iter}
        )
        es.optimize(self.cost_func)
        self.net.calculate_multiple_output(self.x_test)
        test = self.net.backpropagation(self.exp)
        # print(es.result)
        return es.result.xbest, es.result

    def get_values_for_X(self, X):
        Y = []
        for x in X:
            Y.append(float(self.net.calculate_output(x)))
        return Y


def extra_sin_function(x):
    return (np.sin(x)) * 5


def linear_function(x):
    return x * 0.4


def task_function(x):
    val = (x**2) * np.sin(x)
    val += 100 * np.sin(x) * np.cos(x)
    return val


def multiple_test(
    neural_net, function, char_size, nr_of_samples, max_iterations, population, sigma
):
    # first_ws_bs = get_flattened_ws_bs(neural_net.layers)

    x_values = np.linspace(char_size[0], char_size[1], 1000)
    y_values = function(x_values)
    plt.plot(x_values, y_values, label="function")

    # neural_net.update_with_flattened_w_and_b(first_ws_bs)

    results_cls = Multiple_evaluation(
        function,
        neural_net,
        char_size,
        nr_of_samples,
    )

    resultx, result = results_cls.optimise_with_evolutions(
        max_iterations,
        population,
        sigma,
    )
    x_values = np.linspace(char_size[0], char_size[1], nr_of_samples)
    print(f"sigma:{sigma}, pop:{population} = result {result.fbest}")
    neural_net.update_with_flattened_w_and_b(resultx)
    y_best_neural_network = results_cls.get_values_for_X(x_values)
    label = "sigma=" + str(sigma) + " pop=" + str(population)
    plt.plot(x_values, y_best_neural_network, "r", label=label)
    # plt.plot(x_values, y_best_neural_network, "r+")

    x_pred = np.linspace(char_size[0], char_size[1], nr_of_samples)
    y_pred = []
    for one in x_pred:
        y_pred.append(neural_net.calculate_output(one)[0][0])
    plt.plot(x_pred, y_pred, "go")

    plt.legend()
    title = (
        "Neural network function aproximation (samples:"
        + str(nr_of_samples)
        + " popsize:"
        + str(population)
        + " iterations:"
        + str(max_iterations)
        + ")"
    )
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    fs = Activation_function("logistic")
    fl = BaseLayerFunction()

    l1 = LayerBase(5, fl)
    l2 = LayerBase(10, fs)
    l3 = LayerBase(10, fs)
    l4 = LayerBase(5, fl)
    l_out = LayerBase(1, fl)

    nr_of_samples = 50
    imax = 10
    population_size = 50
    sigma = 0.75
    char_size = (-5, 5)

    opt_function = task_function
    n_test = Neural_net([l1, l2, l3, l4, l_out], 1)

    multiple_test(
        n_test, opt_function, char_size, nr_of_samples, imax, population_size, sigma
    )
