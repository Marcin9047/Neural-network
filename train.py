from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import BaseNeuralNetwork, CostFunction
from layer_functions import SigmoidFunction, BaseLayerFunction
from matplotlib import pyplot as plt
import cma


def testing_func(x):
    val = (x**2) * np.sin(x)
    val += 100 * np.sin(x) * np.cos(x)
    return val


def optimise_with_evolutions(
    emulate_func,
    n_m: BaseNeuralNetwork,
    limits,
    nr_per_iter: int,
    max_iter=100000,
    popsize=1000,
    sigma0=0.1,
):
    cf = CostFunction()

    def cost_func(params: array):
        n_m.update_with_flattened_w_and_b(params)
        x_list = np.linspace(limits[0], limits[1], nr_per_iter)
        y_list = emulate_func(x_list)
        y_pred = []
        cost = 0
        for x in x_list:
            y_pred.append(n_m.calculate_output(x))
        cost += cf.get_float_cost(y_list, np.array(y_pred))
        return cost / nr_per_iter

    params_init = n_m.get_flattened_ws_bs()

    es = cma.CMAEvolutionStrategy(
        params_init, sigma0, {"popsize": popsize, "maxiter": max_iter}
    )
    es.optimize(cost_func)
    print(es.result)
    return es.result.xbest, es.result


def get_values_for_X(X, nm: BaseNeuralNetwork):
    Y = []
    for x in X:
        Y.append(float(nm.calculate_output(x)))
    return Y


def extra_sin_function(x):
    return np.sin(x) ** 3


def linear_function(x):
    return x * 0.4


if __name__ == "__main__":
    fs = SigmoidFunction()
    fl = BaseLayerFunction()
    l1 = LayerBase(5, 1, fl)
    l2 = LayerBase(10, 5, fs)
    l3 = LayerBase(10, 10, fs)
    l4 = LayerBase(5, 10, fl)
    # l5 = LayerBase(5, 5, f)
    l_out = LayerBase(1, 5, fl)
    n_manage = BaseNeuralNetwork([l1, l2, l3, l4, l_out])
    first_ws_bs = n_manage.get_flattened_ws_bs()
    function_to_optimise = extra_sin_function

    x_plot = np.linspace(-7, 7, 1000)
    y_function = function_to_optimise(x_plot)
    plt.plot(x_plot, y_function, label="function")

    nr_of_samples = 10
    max_iterations = 10
    population_size = 255
    # sigma = 0.5

    for sigma_from_list in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5]:
        nm = BaseNeuralNetwork([l1, l2, l3, l4, l_out])
        a = nm.get_flattened_ws_bs()
        nm.update_with_flattened_w_and_b(first_ws_bs)
        sigma = sigma_from_list
        resultx, result = optimise_with_evolutions(
            function_to_optimise,
            nm,
            (-1, 4),
            nr_of_samples,
            max_iterations,
            population_size,
            sigma,
        )
        # print(result)
        print(f"sigma:{sigma}, pop:{population_size} = result {result.fbest}")
        n_manage.update_with_flattened_w_and_b(result)
        y_best_neural_network = get_values_for_X(x_plot, n_manage)
        label = "weights sigma=" + str(sigma) + "pop=" + str(population_size)
        plt.plot(x_plot, y_best_neural_network, label=label)

    plt.legend()
    plt.show()
