from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import Neural_net, CostFunction
from layer_functions import (
    SigmoidFunction,
    BaseLayerFunction,
    ReluFunction,
    TanHFunction,
)
from matplotlib import pyplot as plt
import cma
from type_converters import *
from n_n_gradient_descent import (
    GradientSolverParams,
    GradientSolverResults,
    gradient_descent_with_initial_step_optimisation,
)
from n_n_evolutionary_algoritm import (
    EvolutionarySolver,
    EvolutionarySolverParams,
    EvolutionarySolverResults,
)


def extra_sin_function(x):
    return (np.sin(x)) * 5


def linear_function(x):
    return x * 0.4


def task_function(x):
    val = (x**2) * np.sin(x)
    val += 100 * np.sin(x) * np.cos(x)
    return val


def create_2_mid_layer_nn(
    nr_of_neurons: int,
    activation_functions: Tuple[BaseLayerFunction, BaseLayerFunction],
    output_function: BaseLayerFunction,
):
    l1 = LayerBase(nr_of_neurons, activation_functions[0])
    l2 = LayerBase(nr_of_neurons, activation_functions[1])

    l_out = LayerBase(1, output_function)

    return Neural_net([l1, l2, l_out], 1)


if __name__ == "__main__":
    from tqdm import tqdm
    from train import task_function
    from scipy import optimize
    from copy import deepcopy

    train_size = 200
    X = np.linspace(-10, 10, train_size)
    Y = task_function(X)
    cf = CostFunction()

    fs = SigmoidFunction()
    frl = ReluFunction()
    ftan = TanHFunction()
    fl = BaseLayerFunction()

    iter_number = 1000
    evo_to_grad_scalar = 5
    nr_of_neurons = 10
    mid_func = ftan

    n_test = create_2_mid_layer_nn(10, (mid_func, mid_func), fl)

    original_preds = [float(i) for i in n_test.calculate_output_for_many_values(X)]
    spg = GradientSolverParams(iter_number, X, Y, cf)
    spe = EvolutionarySolverParams((iter_number / evo_to_grad_scalar), X, Y, cf)
    srg = gradient_descent_with_initial_step_optimisation(deepcopy(n_test), spg)
    es = EvolutionarySolver(deepcopy(n_test), spe)
    sre = es.optimise_with_evolutions()

    n_g = srg.neural_network
    print("gradient", srg.best_cost)
    n_g.update_with_weights(srg.best_weights)
    n_g.update_with_biases(srg.best_biases)
    plt.plot(srg.cost_hist, label="cost Gradient descent")

    n_e = sre.neural_network
    print("Evolutionary", sre.best_cost)
    n_e.update_with_weights(sre.best_weights)
    n_e.update_with_biases(sre.best_biases)
    # plt.plot(sre.cost_hist, label="cost Evolutionary algoritm")
    # plt.plot(gradient_note["beta_history"], label="beta")
    plt.legend()
    plt.semilogy()
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("Function Aproximation training")
    plt.show()

    plt.plot(X, Y, label="function")
    plt.plot(
        X,
        [float(i) for i in n_g.calculate_output_for_many_values(X)],
        label="gradient descent",
    )
    plt.plot(
        X,
        [float(i) for i in n_e.calculate_output_for_many_values(X)],
        label="evolutionary algoritm",
    )
    plt.plot(
        X,
        original_preds,
        label="neural net before training",
    )

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function Aproximation")
    plt.show()
