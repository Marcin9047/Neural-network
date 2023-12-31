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


def descent_arrays(old_arrays: List[array], arrays_deriv: List[array], B: float):
    new_weights = []
    for i in range(len(old_arrays)):
        new_weights.append((old_arrays[i] - (B * arrays_deriv[i])))
    return new_weights


def compose_many_array_lists(array_lists: List[List[array]]) -> List[array]:
    new_arrays = []
    nr_of_d = len(array_lists)
    rel_d = len(delete_zero_array_lists(array_lists))
    for i in range(len(array_lists[0])):
        comp_array = 0
        for j in range(nr_of_d):
            comp_array += array_lists[j][i]
        new_arrays.append((comp_array / rel_d))
    return new_arrays


def delete_zero_array_lists(array_lists: List[List[array]]) -> List[List[array]]:
    ar2 = []
    for array_l in array_lists:
        delarr = True
        for array_fl in array_l:
            if np.sum(array_fl) != 0:
                delarr = False
        if not delarr:
            ar2.append(array_l)
    return ar2


def gradient_descent(
    B, learn_rate, iter, neural_network: Neural_net, X, Y, cf: CostFunction
):
    cost_hist = []
    B_descenting = B
    best_stuff = None
    best_cost = None
    for i in tqdm(range(iter), leave=False):
        cost_sum = 0
        dweights_list = []
        dbiases_list = []
        for ts in range(len(X)):
            nout = neural_network.calculate_output(X[ts])
            dC = cf.get_deriv_cost_to_a_output(Y[ts], nout)
            cost_sum += cf.get_float_cost(Y[ts], nout)
            dws, dbs = neural_network.backpropagate(dC)
            dweights_list.append(dws)
            dbiases_list.append(dbs)
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
        we_comp = compose_many_array_lists(dweights_list)
        bs_comp = compose_many_array_lists(dbiases_list)
        new_weights = descent_arrays(
            get_weights(neural_network.layers),
            we_comp,
            B_descenting,
        )
        new_biases = descent_arrays(
            get_biases(neural_network.layers),
            bs_comp,
            B_descenting,
        )

        neural_network.update_with_weights(new_weights)
        neural_network.update_with_biases(new_biases)
        B_descenting = B_descenting * learn_rate
    gradient_note = {
        "neural_net": neural_network,
        "best_cost": best_cost,
        "cost_hist": cost_hist,
        "nr_of_iter": iter,
        "ending_B": B_descenting,
        "init_b": B,
        "learn_rate": learn_rate,
        "best_weights_biases": best_stuff,
    }
    return neural_network, cost_hist


import math


def calculate_training_initial_best_angle(cost_hist):
    max_score = 1000

    ba_s = 0.5
    aa_s = 2
    ea_s = 1
    nr_of_evals = ba_s + aa_s + ea_s
    max_angle_val = -1.5707963266948965
    scal = (max_score / nr_of_evals) / max_angle_val
    best_angle = None
    avg_angle = 0
    li = 0

    for i in range(len(cost_hist) - 1):
        li += 1
        val_0 = cost_hist[i]
        val_1 = cost_hist[i + 1]
        ang = math.atan2(val_1 - val_0, 100)
        avg_angle += ang
        if best_angle is None:
            best_angle = ang
        elif best_angle > ang:
            best_angle = ang
    end_angle = math.atan2(cost_hist[-1] - cost_hist[0], 100 * (len(cost_hist)))
    avg_angle = avg_angle / li

    score = (ba_s * best_angle + aa_s * avg_angle + ea_s * end_angle) * scal

    note = {
        "score": score,
        "best_angle": best_angle,
        "end_angle": end_angle,
        "avg_angle": avg_angle,
    }
    return note


def compare_initial_settings(B_list, lr_list, iter_testing, nt: Neural_net):
    from copy import deepcopy

    tests = []
    for B in tqdm(B_list):
        for lr in lr_list:
            n_test = deepcopy(nt)
            e, cost_hist = gradient_descent(B, lr, iter_testing, n_test, X, Y, cf)
            angle_note = calculate_training_initial_best_angle(cost_hist)
            if angle_note["best_angle"] != 0:
                test_note = {
                    "B": B,
                    "lr": lr,
                    **angle_note,
                    "c_hist": cost_hist,
                }
            else:
                test_note = {"B": B, "lr": lr, **angle_note}
            if len(tests) == 0:
                tests.append(test_note)
            elif tests[0]["score"] < test_note["score"]:
                tests.insert(0, test_note)
            else:
                tests.append(test_note)
    return tests


if __name__ == "__main__":
    from tqdm import tqdm
    from train import task_function

    train_size = 50
    X = np.linspace(-5, 5, train_size)
    Y = task_function(X)
    test_iter = 25
    fs = SigmoidFunction()
    frl = BaseLayerFunction()
    ftan = TanHFunction()

    l1 = LayerBase(5, frl)
    l2 = LayerBase(10, fs)
    l3 = LayerBase(10, fs)
    l35 = LayerBase(10, fs)
    l4 = LayerBase(5, frl)
    l_out = LayerBase(1, frl)

    n_test = Neural_net([l1, l2, l3, l35, l4, l_out], 1)
    B_list = np.logspace(-20, -1, 50)
    original_preds = [float(i) for i in n_test.calculate_output_for_many_values(X)]
    cf = CostFunction()
    cs = compare_initial_settings(
        B_list,
        [0.99, 0.99999],
        test_iter,
        n_test,
    )
    for c in cs:
        print(c)
    if cs[0]["score"] == 0:
        raise ValueError("shit gatcha")
    best_b = cs[0]["B"]
    best_lr = cs[0]["lr"]
    print("best:", cs[0])
    print(best_b, best_lr)
    neural_network, cost_hist = gradient_descent(
        best_b, best_lr, 1000, n_test, X, Y, cf
    )
    # print(calculate_training_initial_best_angle(cost_hist))
    plt.plot(cost_hist)
    plt.semilogy()
    plt.show()
    plt.plot(X, Y, label="function")
    plt.plot(
        X,
        [float(i) for i in n_test.calculate_output_for_many_values(X)],
        label="neural net",
    )
    plt.plot(
        X,
        original_preds,
        label="neural net_before training",
    )

    plt.legend()
    plt.show()
