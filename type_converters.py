from numpy import array
import numpy as np
from typing import Callable, Tuple, List


def convert_w_to_list(w: array) -> array:
    list_w_val = np.ravel(w)
    return list_w_val


def convert_list_to_w(list_w: array, w_size: Tuple[int, int]) -> array:
    # zmienia vector wartości wag na macierz wag
    w = list_w.reshape(w_size)
    return w


def get_flattened_weights(layers):
    # zwraca vector wszystkich płaskich wag dla wszystkich warstw
    full_weights = []
    for layer in layers:
        full_weights.append(layer.get_flat_weights_vector())
    return np.concatenate(full_weights)


def get_flattened_bias(layers):
    # zwraca vector wszystkich płaskich biasów dla wszystkich warstw
    full_bias = []
    for layer in layers:
        full_bias.append(layer.b)
    return np.concatenate(full_bias)


def get_flattened_ws_bs(layers):
    # Zwraca wektor wszystkich wag i biasów dla wszystkich warstw (wykorzystywany do optymalizacji parametrów całego neural network)
    ws = get_flattened_weights(layers)
    bs = get_flattened_bias(layers)
    all_s = []
    for w in ws:
        all_s.append(w)
    for b in bs:
        all_s.append(b)
    return array(all_s)


def update_with_flattened_bias(layers, flattened_bias: array):
    # zmienia biasy warstw na te podane w wektorze biasów wszystkich warstw
    flat_sizes = []
    for layer in layers:
        flat_sizes.append(layer.b_size)
    vector_list = np.split(flattened_bias, np.cumsum(flat_sizes)[:-1])
    for i in range(len(layers)):
        layers[i].update_b_with_flat(vector_list[i])


def update_with_flattened_w_and_b(layers, flattened_w_b: array):
    flat_w_size = len(get_flattened_weights(layers))
    # zmienia biasy i wagi warstw na te podane w wektorze biasów i wag wszystkich warstw
    flat_ws = flattened_w_b[:flat_w_size]
    flat_bs = flattened_w_b[flat_w_size:]
    update_with_flattened_bias(layers, flat_bs)
    update_with_flattened_weights(layers, flat_ws)


def update_with_flattened_weights(layers, flattened_ws: array):
    # zmienia wagi warstw na te podane w wektorze wag wszystkich warstw
    flat_sizes = []
    for layer in layers:
        flat_sizes.append(layer.full_size_of_w)
    vector_list = np.split(flattened_ws, np.cumsum(flat_sizes)[:-1])
    for i in range(len(layers)):
        layers[i].update_w_with_flat(vector_list[i])
