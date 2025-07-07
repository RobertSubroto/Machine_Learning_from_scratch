# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 21:44:48 2025

@author: Bobby Subroto
"""

import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def feedforward(x, weights):
    """
    Perform a full forward pass, caching activations and pre-activations.

    Parameters:
    - x: input vector (shape: input_size,)
    - weights: list of weight matrices

    Returns:
    - activations: list of activations per layer (a^0 to a^L)
    - zs: list of pre-activations z^1 to z^L
    """
    activations = [x]
    zs = []
    a = x
    for W in weights:
        z = W @ a
        a = relu(z)
        zs.append(z)
        activations.append(a)
    return activations, zs

def dC_da_L(a_L, y):
    """
    Gradient of MSE cost w.r.t. final activation a^L
    """
    return 2 * (a_L - y)

def da_dz(z):
    """
    Derivative of ReLU activation w.r.t. z
    Returns a diagonal matrix
    """
    return np.diag(relu_derivative(z))

def differential_matrix_tracker(weights, x, y):
    """
    Tracks the backpropagated gradient vector ∂C/∂z^l at each layer (in reverse) for one input output 
    pair x and y.

    Returns:
    - list of ∂C/∂z^l for l = L to 1
    """
    activations, zs = feedforward(x, weights)
    L = len(weights)

    # Initial: ∂C/∂a^L → ∂C/∂z^L
    grad = da_dz(zs[-1]) @ dC_da_L(activations[-1], y)
    grads = [grad]

    # Backpropagate ∂C/∂z^(l) = da_dz(z^(l)) @ W^(l+1).T @ ∂C/∂z^(l+1)
    for l in reversed(range(1, L)):
        dz = da_dz(zs[l-1])
        W_next = weights[l]
        grad = dz @ W_next.T @ grads[-1]
        grads.append(grad)

    grads.reverse()  # So grads[0] = ∂C/∂z^1, ..., grads[-1] = ∂C/∂z^L
    return grads

def total_differential_matrix_tracker_weights(weights, X, Y):
    """
    Accumulates the total weight gradients ∂C/∂W^l over the dataset.

    Returns:
    - list of accumulated weight gradient matrices, one per layer
    """
    total_grads = [np.zeros_like(W) for W in weights]

    for i in range(X.shape[0]):
        grads = differential_matrix_tracker(weights, X[i], Y[i])  # ∂C/∂z for each layer
        activations, zs = feedforward(X[i], weights)              # lists of activations and pre-activations
        
        for l in range(len(weights)):
            delta = grads[l][:, None]           # (n_out, 1) column vector
            a_prev = activations[l][None, :]   # (1, n_in) row vector
            total_grads[l] += delta @ a_prev   # outer product, shape (n_out, n_in)
    
    return total_grads
