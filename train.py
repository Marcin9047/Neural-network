from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import BaseNeuralNetwork, CostFunction
from layer_functions import SigmoidFunction, LayerFunction


def testing_func(x):
    val = (x**2) * np.sin(x)
    val += 100 * np.sin(x) * np.cos(x)
    return val


def optimise_with_scipy(
    emulate_func,
    n_m: BaseNeuralNetwork,
    limits,
    nr_per_iter: int,
    max_iter=100000,
    popsize=1000,
    sigma0=0.1,
):
    cf = CostFunction()
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
    es = cma.CMAEvolutionStrategy(
        params_init, sigma0, {"popsize": popsize, "maxiter": max_iter}
    )
    es.optimize(cost_func)
    print(es.result)
    return es.result.xbest, cost_history, n_m


def get_values_for_X(X, nm: BaseNeuralNetwork):
    Y = []
    for x in X:
        Y.append(float(nm.calculate_output(x)))
    return Y


def sin_function(x):
    return np.sin(x)


if __name__ == "__main__":
    fs = SigmoidFunction()
    fl = LayerFunction()
    l1 = LayerBase(5, 1, fs)
    l2 = LayerBase(10, 5, fl)
    l3 = LayerBase(10, 10, fl)
    l4 = LayerBase(5, 10, fs)
    # l5 = LayerBase(5, 5, f)
    l_out = LayerBase(1, 5, fl)

    n_manage = BaseNeuralNetwork([l1, l2, l3, l4, l_out])
    first_ws_bs = n_manage.get_flattened_ws_bs()
    result, cost_his, n_m = optimise_with_scipy(
        sin_function,
        n_manage,
        (-1, 4),
        40,
        5000,
        200,
    )

    print(result)
    best_ws_bs = result

    from matplotlib import pyplot as plt

    n_manage.update_with_flattened_w_and_b(best_ws_bs)
    x_plot = np.linspace(-7, 7, 1000)
    y_perf = sin_function(x_plot)

    y_best = get_values_for_X(x_plot, n_manage)
    # n_manage.update_with_flattened_w_and_b(first_ws_bs)
    # y_first = get_values_for_X(x_plot, n_manage)
    plt.plot(x_plot, y_perf, label="function")
    # plt.plot(x_plot, y_first, label="random weights")
    plt.plot(x_plot, y_best, label="weigths trained")
    plt.legend()
    # # plt.semilogy()
    plt.show()
