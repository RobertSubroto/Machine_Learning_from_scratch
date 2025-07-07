# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 18:09:45 2025

@author: Bobby Subroto
"""

import numpy as np
import NN_backpropagation as nnback



def C(X, Y, weights):
    total_cost = 0.0
    for x, y in zip(X, Y):
        a, _ = nnback.feedforward(x, weights)
        total_cost += np.sum((a[-1] - y) ** 2)
    return total_cost / len(X)


def gradient_descent(weights, learning_rate, number_of_steps, X, Y):
    weights_min = [W.copy() for W in weights]
    for _ in range(number_of_steps):
        gradients = nnback.total_differential_matrix_tracker_w(weights_min, X, Y)  # Compute once
        for j in range(len(weights_min)):
            weights_min[j] -= learning_rate * gradients[j]  # Update weights
    return weights_min



