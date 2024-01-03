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
import random
import math


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
    var_step_size: bool = False,
    learn_rate: float = 1,
    B_init: float = None,
    sample_size=None,
) -> dict:
    cost_hist = []
    if B_init is None:
        B_descenting = B
    else:
        B_descenting = B_init
    best_stuff = None
    best_cost = None
    beta_history = []
    t = 1
    for i in tqdm(range(iter), leave=False):
        cost_sum = 0
        if sample_size is not None:
            X_s, Y_s = choose_n_random_samples_from_training_data(X, Y, sample_size)
        else:
            X_s = X
            Y_s = Y
        gradient_w, gradient_b, cost_sum = get_gradients_cost_per_sample(
            X_s, Y_s, neural_network, cf
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
        if var_step_size:
            B_descenting = adapt_beta(
                neural_network, gradient_w, gradient_b, B_descenting, B, cf, X_s, Y_s
            )
        new_weights = descent_arrays(
            get_weights(neural_network.layers),
            gradient_w,
            B_descenting,
        )
        new_biases = descent_arrays(
            get_biases(neural_network.layers),
            gradient_b,
            B_descenting,
        )

        neural_network.update_with_weights(new_weights)
        neural_network.update_with_biases(new_biases)
        B_descenting = B_descenting * learn_rate
        beta_history.append(B_descenting)
    gradient_note = {
        "neural_network": neural_network,
        "best_cost": best_cost,
        "cost_hist": cost_hist,
        "nr_of_iter": iter,
        "ending_B": B_descenting,
        "init_b": B,
        "learn_rate": learn_rate,
        "best_weights_biases": best_stuff,
        "beta_history": beta_history,
    }
    return gradient_note


def calculate_training_initial_best_angle(cost_hist: List[float]) -> dict:
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


def compare_initial_settings(
    B_list: List[float],
    iter_testing: int,
    nt: Neural_net,
    sample_size: int = None,
    learn_rate: float = 1,
    vary_step=False,
) -> List[dict]:
    from copy import deepcopy

    tests = []
    for B in tqdm(B_list):
        n_test = deepcopy(nt)
        grad_note = gradient_descent(
            B,
            iter_testing,
            n_test,
            X,
            Y,
            cf,
            sample_size=sample_size,
            var_step_size=vary_step,
            learn_rate=learn_rate,
        )
        angle_note = calculate_training_initial_best_angle(grad_note["cost_hist"])
        if angle_note["best_angle"] != 0:
            test_note = {
                "B": B,
                "lr": lr,
                **angle_note,
                "c_hist": grad_note["cost_hist"],
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


# def compare_initial_settings_var_step(
#     init_B_list: List[float],
#     iter_testing: int,
#     nt: Neural_net,
#     sample_size: int,
#     learn_rate: float = 1,
#     vary_step=False,
# ):
#     from copy import deepcopy

#     tests = []
#     for B in tqdm(B_list):
#         n_test = deepcopy(nt)
#         grad_note = gradient_descent(
#             B,
#             iter_testing,
#             n_test,
#             X,
#             Y,
#             cf,
#             sample_size,
#             var_step_size=vary_step,
#             learn_rate=learn_rate,
#         )
#         angle_note = calculate_training_initial_best_angle(grad_note["cost_hist"])
#         if angle_note["best_angle"] != 0:
#             test_note = {
#                 "B": B,
#                 "lr": lr,
#                 **angle_note,
#                 "c_hist": grad_note["cost_hist"],
#             }
#         else:
#             test_note = {"B": B, "lr": lr, **angle_note}
#         if len(tests) == 0:
#             tests.append(test_note)
#         elif tests[0]["score"] < test_note["score"]:
#             tests.insert(0, test_note)
#         else:
#             tests.append(test_note)
#     return tests


def choose_n_random_samples_from_training_data(
    X: List[array], Y: List[array], n: int
) -> Tuple[List[array], List[array]]:
    size = len(X)
    random_indices = random.sample(range(size), n)

    # Extract the corresponding elements from X_list and y_list
    X_samples = [X[i] for i in random_indices]
    y_samples = [Y[i] for i in random_indices]
    return X_samples, y_samples


def adapt_beta(
    neural_network: Neural_net,
    gradient_w: List[array],
    gradient_b: List[array],
    t: float,
    B: float,
    cf: CostFunction,
    X: List[array],
    Y: List[array],
) -> float:
    cost_pure = get_cost_per_sample(X, Y, neural_network, cf)
    init_w = get_weights(neural_network.layers)
    fullgrad_w = descent_arrays(
        init_w,
        gradient_w,
        1,
    )
    init_b = get_biases(neural_network.layers)
    fullgrad_b = descent_arrays(
        init_b,
        gradient_b,
        1,
    )
    neural_network.update_with_weights(fullgrad_w)
    neural_network.update_with_biases(fullgrad_b)
    cost_fullgrad = get_cost_per_sample(X, Y, neural_network, cf)
    w_grad_full = [convert_w_to_list(fg) for fg in fullgrad_w]
    b_grad_full = [convert_w_to_list(fg) for fg in fullgrad_b]
    n_w = np.linalg.norm(np.concatenate(w_grad_full), ord=2)
    n_b = np.linalg.norm(np.concatenate(b_grad_full), ord=2)
    # norm_grad = np.linalg.norm([n_w, n_b], ord=2)
    if cost_fullgrad > (cost_pure - ((t / 2) * n_w)):
        t = B * t
    neural_network.update_with_weights(init_w)
    neural_network.update_with_biases(init_b)
    return t


if __name__ == "__main__":
    from tqdm import tqdm
    from train import task_function
    from scipy import optimize

    ending_beta_scalar = 1
    train_size = 200
    X = np.linspace(-10, 10, train_size)
    Y = task_function(X)
    test_iter = 10
    primary_tries = 50
    secondary_tries = 100
    rel_train = 10000
    lr = np.power(ending_beta_scalar, (1 / rel_train))
    # B = 0.9
    fs = SigmoidFunction()
    frl = ReluFunction()
    ftan = TanHFunction()
    fl = BaseLayerFunction()

    # l1 = LayerBase(5, ftan)
    l2 = LayerBase(100, ftan)
    l3 = LayerBase(100, ftan)
    l_out = LayerBase(1, fl)

    n_test = Neural_net([l2, l3, l_out], 1)
    # n_test = Neural_net([l2, l_out], 1)

    B_list = np.logspace(-20, -1, 50)
    original_preds = [float(i) for i in n_test.calculate_output_for_many_values(X)]
    cf = CostFunction()
    cs = compare_initial_settings(
        B_list,
        test_iter,
        n_test,
        learn_rate=lr,
    )
    if cs[0]["score"] == 0:
        raise ValueError("shit gatcha")
    best_b = cs[0]["B"]
    print(best_b)
    amplitude = 0
    for i in range(len(B_list)):
        if best_b == B_list[i]:
            v1 = B_list[i - 1]
            v2 = B_list[i + 1]
    second_t = np.linspace(v1, v2, (secondary_tries - 1))
    second_t = [*list(second_t), best_b]
    cs = compare_initial_settings(
        second_t,
        test_iter,
        n_test,
        learn_rate=lr,
    )
    best_b = cs[0]["B"]
    print(best_b)

    gradient_note = gradient_descent(
        best_b,
        rel_train,
        n_test,
        X,
        Y,
        cf,
        learn_rate=lr,
    )
    # print(calculate_training_initial_best_angle(cost_hist))
    # gradient_note = {
    #     "neural_network": neural_network,
    #     "best_cost": best_cost,
    #     "cost_hist": cost_hist,
    #     "nr_of_iter": iter,
    #     "ending_B": B_descenting,
    #     "init_b": B,
    #     "learn_rate": learn_rate,
    #     "best_weights_biases": best_stuff,
    #     "beta_history": beta_history,
    n_test = gradient_note["neural_network"]
    best_ws_bs = gradient_note["best_weights_biases"]
    n_test.update_with_weights(best_ws_bs[0])
    n_test.update_with_biases(best_ws_bs[1])
    plt.plot(gradient_note["cost_hist"], label="cost")
    # plt.plot(gradient_note["beta_history"], label="beta")
    plt.legend()
    plt.semilogy()
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("Function Aproximation - Gradient descent")
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
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function Aproximation - Gradient descent")
    plt.show()
