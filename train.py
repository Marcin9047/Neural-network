from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import Neural_net, CostFunction
from layer_functions import SigmoidFunction, BaseLayerFunction
from matplotlib import pyplot as plt
import cma
from type_converters import *


def optimise_with_evolutions(
    emulate_func,
    n_m: Neural_net,
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

    params_init = get_flattened_ws_bs(n_m.layers)

    es = cma.CMAEvolutionStrategy(
        params_init, sigma0, {"popsize": popsize, "maxiter": max_iter}
    )
    es.optimize(cost_func)
    # print(es.result)
    return es.result.xbest, es.result


def get_values_for_X(X, nm: Neural_net):
    Y = []
    for x in X:
        Y.append(float(nm.calculate_output(x)))
    return Y


def extra_sin_function(x):
    return (np.sin(x)) * 5


def linear_function(x):
    return x * 0.4


def task_function(x):
    val = (x**2) * np.sin(x)
    val += 100 * np.sin(x) * np.cos(x)
    return val


if __name__ == "__main__":
    i = 3
    fs = SigmoidFunction()
    fl = BaseLayerFunction()
    l1 = LayerBase(5, fl)
    l2 = LayerBase(10, fs)
    l3 = LayerBase(10, fs)
    l4 = LayerBase(5, fl)
    # l5 = LayerBase(5, 5, f)
    l_out = LayerBase(1, fl)
    n_manage = Neural_net([l1, l2, l3, l4, l_out], 1)
    first_ws_bs = get_flattened_ws_bs(n_manage.layers)
    # n_manage.up
    function_to_optimise = linear_function
    size = (-5, 5)
    x_plot = np.linspace(size[0], size[1], 1000)
    y_function = function_to_optimise(x_plot)

    nr_of_samples = 100
    max_iterations = 250
    # population_size = 255
    # sigma = 0.5
    for pop_size in [10, 25, 50, 100, 200, 255]:
        plt.plot(x_plot, y_function, label="function")
        for sigma_from_list in [0.01, 0.1, 0.5, 0.9, 1.1, 1.5]:
            sigma = sigma_from_list
            population_size = pop_size
            test_note = {
                "sigma": sigma,
                "pop_size": population_size,
                "iter": max_iterations,
                "samples": nr_of_samples,
            }
            nm = Neural_net([l1, l2, l3, l4, l_out], 1)
            nm.update_with_flattened_w_and_b(first_ws_bs)

            resultx, result = optimise_with_evolutions(
                function_to_optimise,
                nm,
                size,
                nr_of_samples,
                max_iterations,
                population_size,
                sigma,
            )
            # print(result)
            print(f"sigma:{sigma}, pop:{population_size} = result {result.fbest}")
            test_note["result"] = result.fbest
            test_note["flatwsbs"] = list(resultx)
            n_manage.update_with_flattened_w_and_b(resultx)
            y_best_neural_network = get_values_for_X(x_plot, n_manage)
            label = "weights sigma=" + str(sigma) + "pop=" + str(population_size)
            plt.plot(x_plot, y_best_neural_network, label=label)

        plt.legend()
        # plt.show()
        title = (
            "Neural network function aproximation (samples:"
            + str(nr_of_samples)
            + " popsize:"
            + str(population_size)
            + " iterations:"
            + str(max_iterations)
            + ")"
        )
        plt.title(title)
        path = (
            "figures/"
            + str(i)
            + "/sam"
            + str(nr_of_samples)
            + "_pop"
            + str(population_size)
            + "_iter"
            + str(max_iterations)
            + ".png"
        )
        plt.savefig(path)
        plt.close()
