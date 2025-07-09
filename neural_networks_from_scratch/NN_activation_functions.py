# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 23:16:06 2025

@author: Bobby Subroto
"""

import numpy as np



# ReLU function
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid_second_derivative(x):
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2 * s)




# Python function for the activation function
def activation_function(function, level):
    if function == 'ReLU':
        if level == 0:
            return relu
        elif level == 1:
            return relu_derivative
    
    if function == 'sigmoid':
        if level == 0:
            return sigmoid 
        elif level == 1:
            return sigmoid_derivative
        elif level == 2:
            return sigmoid_second_derivative
            


a = activation_function('ReLU', 0)(-5)




