# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 18:09:45 2025

@author: Bobby Subroto
"""

import numpy as np
import NN_backpropagation as nnback



def cost(X, Y, weights, activation_function):
    predictions = np.array([nnback.feedforward(x, weights, activation_function)[0][-1] for x in X])
    return np.mean(np.sum((predictions - Y)**2, axis=1))  # MSE across all rows



def gradient_descent(weights, learning_rate, number_of_steps, X, Y, activation_function):
    weights_min = [W.copy() for W in weights]
    for _ in range(number_of_steps):
        gradients = nnback.gradient_weights_matrix_form(weights_min, X, Y, activation_function)  # Compute once
        for j in range(len(weights_min)):
            weights_min[j] -= learning_rate * gradients[j]  # Update weights
    return weights_min



