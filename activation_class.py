import math
import numpy as np


class activation_function:
    def __init__(self, activation_type: str):
        self.type = activation_type

    def get_value(self, x):
        if self.type == "logistic":
            return 1 / (1 + math.exp(-x))
        elif self.type == "binary step":
            return np.where(x < 0, 0, 1)
        elif self.type == "ReLU":
            return np.where(x < 0, 0, x)
        elif self.type == "TanH":
            return 2 / (1 + math.exp(-2 * x)) - 1
