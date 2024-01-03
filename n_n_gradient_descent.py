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
from type_converters import *
import math
from tqdm import tqdm


class GradientSolverResults:
    def __init__(self):
        self.neural_network = None
        self.best_cost = None
        self.cost_hist = None
        self.nr_of_iter = None
        self.ending_B = None
        self.init_b = None
        self.learn_rate = None
        self.best_weights = None
        self.best_biases = None
        self.beta_history = None


def subtract_gradients_from_arrays(
    old_arrays: List[array], arrays_gradients: List[array], B: float
) -> List[array]:
    new_weights = []
    for i in range(len(old_arrays)):
        new_weights.append((old_arrays[i] - (B * arrays_gradients[i])))
    return new_weights


def compose_many_array_lists(array_lists: List[List[array]]) -> List[array]:
    new_arrays = []
    nr_of_d = len(array_lists)
    for i in range(len(array_lists[0])):
        comp_array = 0
        for j in range(nr_of_d):
            comp_array += array_lists[j][i]
        new_arrays.append((comp_array / nr_of_d))
    return new_arrays


def get_gradients_cost_per_sample(
    X: List[array], Y: List[array], neural_network: Neural_net, cf: CostFunction
) -> Tuple[List[array], List[array], float]:
    dweights_list = []
    dbiases_list = []
    cost_sum = 0
    for ts in range(len(X)):
        nout = neural_network.calculate_output(X[ts])
        dC = cf.get_deriv_cost_to_a_output(Y[ts], nout)
        cost_sum += cf.get_float_cost(Y[ts], nout)
        dws, dbs = neural_network.backpropagate(dC)
        dweights_list.append(dws)
        dbiases_list.append(dbs)
    cost = cost_sum / len(X)
    we_comp = compose_many_array_lists(dweights_list)
    bs_comp = compose_many_array_lists(dbiases_list)
    return (we_comp, bs_comp, cost)


def get_cost_per_sample(
    X: List[array], Y: List[array], neural_network: Neural_net, cf: CostFunction
) -> float:
    cost_sum = 0
    for ts in range(len(X)):
        nout = neural_network.calculate_output(X[ts])
        cost_sum += cf.get_float_cost(Y[ts], nout)
    cost = cost_sum / len(X)
    return cost


def gradient_descent(
    B: float,
    iter: int,
    neural_network: Neural_net,
    X: List[array],
    Y: list[array],
    cf: CostFunction,
    learn_rate: float = 1,
) -> GradientSolverResults:
    cost_hist = []
    B_descenting = B
    best_stuff = None
    best_cost = None
    beta_history = []
    for i in tqdm(range(iter), leave=False):
        gradient_w, gradient_b, cost_sum = get_gradients_cost_per_sample(
            X, Y, neural_network, cf
        )
        cost_hist.append(float(cost_sum))

        if best_cost is None:
            best_cost = cost_sum
            best_stuff = (
                get_weights(neural_network.layers),
                get_biases(neural_network.layers),
            )
        elif cost_sum < best_cost:
            best_cost = cost_sum
            best_stuff = (
                get_weights(neural_network.layers),
                get_biases(neural_network.layers),
            )

        new_weights = subtract_gradients_from_arrays(
            get_weights(neural_network.layers),
            gradient_w,
            B_descenting,
        )
        new_biases = subtract_gradients_from_arrays(
            get_biases(neural_network.layers),
            gradient_b,
            B_descenting,
        )

        neural_network.update_with_weights(new_weights)
        neural_network.update_with_biases(new_biases)
        B_descenting = B_descenting * learn_rate
        beta_history.append(B_descenting)
    sr = GradientSolverResults()
    sr.neural_network = neural_network
    sr.best_cost = best_cost
    sr.cost_hist = cost_hist
    sr.nr_of_iter = iter
    sr.ending_B = B_descenting
    sr.init_b = B
    sr.learn_rate = learn_rate
    sr.best_weights = best_stuff[0]
    sr.best_biases = best_stuff[1]
    sr.beta_history = beta_history
    return sr


def calculate_training_convergence_score(
    cost_hist: List[float],
    max_score: int = 1000,
    best_angle_weight: int = 0.5,
    average_angle_weight: int = 2,
    end_angle_weight: int = 1,
) -> dict:
    nr_of_evals = best_angle_weight + average_angle_weight + end_angle_weight
    max_angle_val = -1.5707963266948965
    scal = (max_score / nr_of_evals) / max_angle_val
    best_angle = None
    avg_angle = 0

    for i in range(len(cost_hist) - 1):
        val_0 = cost_hist[i]
        val_1 = cost_hist[i + 1]
        ang = math.atan2(val_1 - val_0, 100)
        avg_angle += ang
        if best_angle is None:
            best_angle = ang
        elif best_angle > ang:
            best_angle = ang
    end_angle = math.atan2(cost_hist[-1] - cost_hist[0], 100 * (len(cost_hist)))
    avg_angle = avg_angle / (len(cost_hist) - 1)

    score = (
        best_angle_weight * best_angle
        + average_angle_weight * avg_angle
        + end_angle_weight * end_angle
    ) * scal

    note = {
        "score": score,
        "best_angle": best_angle,
        "end_angle": end_angle,
        "avg_angle": avg_angle,
    }
    return note


def compare_initial_settings(
    X: List[array],
    Y: List[array],
    B_list: List[float],
    iter_testing: int,
    nt: Neural_net,
    cf: CostFunction,
    learn_rate: float = 1,
) -> List[dict]:
    from copy import deepcopy

    tests = []
    for B in tqdm(B_list, leave=False):
        n_test = deepcopy(nt)
        sr = gradient_descent(
            B,
            iter_testing,
            n_test,
            X,
            Y,
            cf,
            learn_rate=learn_rate,
        )
        angle_note = calculate_training_convergence_score(sr.cost_hist)
        test_note = {
            "B": B,
            **angle_note,
        }
        if len(tests) == 0:
            tests.append(test_note)
        elif tests[0]["score"] < test_note["score"]:
            tests.insert(0, test_note)
        else:
            tests.append(test_note)
    return tests


class GradientSolverParams:
    def __init__(
        self,
        iteration_nr,
        X,
        Y,
        cost_function: CostFunction,
        learn_rate: int = 1,
        initial_config_iterations: int = 10,
        log_configuration_passes=50,
        lin_configuration_passes=50,
        log_limits: Tuple[int] = (-10, 0),
    ):
        self.iteration_nr = iteration_nr
        self.X = X
        self.Y = Y
        self.cf = cost_function
        self.lr = learn_rate
        self.init_config_iter = initial_config_iterations
        self.log_configuration_passes = log_configuration_passes
        self.lin_configuration_passes = lin_configuration_passes
        self.log_limits = log_limits


def gradient_descent_with_initial_step_optimisation(
    neural_network: Neural_net, sp: GradientSolverParams
) -> GradientSolverResults:
    log_B_list = np.logspace(
        sp.log_limits[0], sp.log_limits[1], sp.log_configuration_passes
    )
    test_iter = sp.init_config_iter
    cs_log = compare_initial_settings(
        sp.X, sp.Y, log_B_list, test_iter, neural_network, sp.cf, learn_rate=sp.lr
    )

    if cs_log[0]["score"] == 0:
        raise ValueError(
            "All configuration tries scored less than 0 in training evaluation"
        )

    best_b = cs_log[0]["B"]
    print(best_b)
    # amplitude = 0
    for i in range(len(log_B_list)):
        if best_b == log_B_list[i]:
            v1 = log_B_list[i - 1]
            v2 = log_B_list[i + 1]
            # amplitude = min(abs(best_b-v1), abs(best_b-v2))
    lin_B_list = np.linspace(v1, v2, (sp.lin_configuration_passes - 1))
    lin_B_list = [*list(lin_B_list), best_b]
    cs_lin = compare_initial_settings(
        sp.X, sp.Y, lin_B_list, test_iter, neural_network, sp.cf, learn_rate=sp.lr
    )
    best_b = cs_lin[0]["B"]
    print(best_b)
    sr = gradient_descent(
        best_b,
        sp.iteration_nr,
        neural_network,
        sp.X,
        sp.Y,
        sp.cf,
        learn_rate=sp.lr,
    )
    return sr
