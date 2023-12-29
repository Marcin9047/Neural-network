from layers_class import LayerBase
from numpy import array
import numpy as np
from typing import Callable, Tuple, List
from neural_net_class import Neural_net, CostFunction
from layer_functions import SigmoidFunction, BaseLayerFunction
from matplotlib import pyplot as plt
import cma
from type_converters import *


def descent_weights(old_weights: List[array], w_deriv: List[array], B: float):
    new_weights = []
    for i in range(len(old_weights)):
        new_weights.append((old_weights[i] - (B * w_deriv[i])))
    return new_weights


def compose_many_gradients(w_derivs: List[List[array]]) -> List[array]:
    new_weights = []
    nr_of_d = len(w_derivs)
    for i in range(len(w_derivs[0])):
        weight = 0
        for j in range(nr_of_d):
            weight += w_derivs[j][i]
        new_weights.append((weight / nr_of_d))
    return new_weights


if __name__ == "__main__":
    from tqdm import tqdm

    train_size = 10
    x = [i for i in range(train_size)]
    y = [(i**2) for i in x]

    fs = SigmoidFunction()
    fl = BaseLayerFunction()

    l1 = LayerBase(5, fl)
    l2 = LayerBase(10, fs)
    l3 = LayerBase(10, fs)
    l4 = LayerBase(5, fl)
    l_out = LayerBase(1, fl)
    n_test = Neural_net([l1, l2, l3, l4, l_out], 1)
    import json

    # with open("n_weights_5_10_10_5_1.json", "r") as f:
    #     n_stuff = json.load(f)
    # # with open("n_weights_5_10_10_5_1.json", "w") as f:
    # #     json.dump(list(get_flattened_weights(n_test.layers)), f)
    # n_test.update_with_flattened_weights(n_stuff)
    print(y)
    print(n_test.calculate_output_for_many_values(x))
    cf = CostFunction()
    B = 0.01
    learn_rate = 0.99
    iter_nr = 24
    cost_hist = []
    value_hist = []
    B_descenting = B
    for i in tqdm(range(iter_nr)):
        d_ws_list = []
        cost_sum = 0
        dweights_list = []
        for t_s in range(train_size):
            nout = n_test.calculate_output(x[t_s])
            value_hist.append(nout)
            dC = cf.get_deriv_cost_to_a_output(y[t_s], nout)
            cost_sum += cf.get_float_cost(y[t_s], nout)
            dweights_list.append(n_test.backpropagate_w(dC))

        cost_hist.append(float(cost_sum))
        # print(dweights_list)
        we_comp = compose_many_gradients(dweights_list)
        print(we_comp)
        new_weights = descent_weights(
            get_weights(n_test.layers),
            we_comp,
            B_descenting,
        )

        n_test.update_with_weights(new_weights)
        B_descenting = B_descenting * learn_rate

    # print(get_weights(n_test.layers))
    # plt.plot(x, y)
    print(y)
    print(n_test.calculate_output_for_many_values(x))
    plt.plot(cost_hist)
    plt.semilogy()
    plt.show()
    plt.plot(x, y, label="function")
    plt.plot(
        x,
        [float(i) for i in n_test.calculate_output_for_many_values(x)],
        label="neural net",
    )

    plt.legend()
    plt.show()
