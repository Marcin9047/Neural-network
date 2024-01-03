from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import Neural_net, CostFunction
from layer_functions import SigmoidFunction, BaseLayerFunction
from matplotlib import pyplot as plt
import cma
from type_converters import *


class EvolutionarySolverResults:
    def __init__(self):
        self.neural_network = None
        self.best_cost = None
        self.cost_hist = None
        self.nr_of_iter = None
        self.best_weights = None
        self.best_biases = None


class EvolutionarySolverParams:
    def __init__(
        self,
        iteration_nr,
        X,
        Y,
        cost_function: CostFunction,
        sigma: float = 0.3,
        popsize: int = 50,
    ):
        self.iteration_nr = iteration_nr
        self.X = X
        self.Y = Y
        self.cf = cost_function
        self.sigma = sigma
        self.popsize = popsize


class EvolutionarySolver:
    def __init__(self, net: Neural_net, sp: EvolutionarySolverParams):
        self.net = net
        self.X = sp.X
        self.Y = sp.Y
        self.cf = sp.cf
        self.sp = sp

    def cost_func(self, params: array):
        self.net.update_with_flattened_w_and_b(params)

        y_pred = []
        for x in self.X:
            y_pred.append(self.net.calculate_output(x)[0][0])
        cost = self.cf.get_cost(self.Y, y_pred)
        val = np.sum(cost) / len(self.X)
        return val

    def optimise_with_evolutions(
        self,
        max_iter=None,
        popsize=None,
        sigma0=None,
    ):
        params_init = get_flattened_ws_bs(self.net.layers)
        if max_iter is None:
            max_iter = self.sp.iteration_nr
        es = cma.CMAEvolutionStrategy(
            params_init, sigma0, {"popsize": popsize, "maxiter": max_iter}
        )
        es.optimize(self.cost_func)
        # print(es.result)
        sr = EvolutionarySolverResults()
        self.net.update_with_flattened_w_and_b(es.result.xbest)
        # self.neural_network = None
        # self.best_cost = None
        # self.cost_hist = None
        # self.nr_of_iter = None
        # self.best_weights = None
        # self.best_biases = None
        sr.best_cost = es.result.fbest
        sr.neural_network = self.net
        sr.best_weights = get_weights(self.net.layers)
        sr.best_biases = get_biases(self.net.layers)
        sr.cost_hist = es.result.evals_best
        print(es.result.evaluations)
        return sr

    def get_values_for_X(self, X):
        Y = []
        for x in X:
            Y.append(float(self.net.calculate_output(x)))
        return Y


if __name__ == "__main__":
    fs = SigmoidFunction()
    fl = BaseLayerFunction()

    l1 = LayerBase(5, fl)
    l2 = LayerBase(10, fs)
    l3 = LayerBase(10, fs)
    l4 = LayerBase(5, fl)
    l_out = LayerBase(1, fl)
    n_test = Neural_net([l1, l2, l3, l4, l_out], 1)
    nr_of_samples = 200
    imax = 1000
    population_size = 50
    sigma = 0.3
    char_size = (-10, 10)

    opt_function = task_function

    multiple_test(
        n_test, opt_function, char_size, nr_of_samples, imax, population_size, sigma
    )
