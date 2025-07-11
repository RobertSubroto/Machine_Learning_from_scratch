# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:42:16 2025

@author: Bobby Subroto
"""

import numpy as np
import NN_activation_functions as nnact
import NN_backpropagation as nnback
import NN_gradient_descent as nngrad



class NeuralNetworks:
    
    def __init__(
        self, X, Y, activation_function = 'ReLU', internal_layers = None, weights_init = None, rate = 0.01, steps = 1000
    ):
        self.X = X
        self.Y = Y
        self.activation_function = activation_function
        if internal_layers is None:
            self.internal_layers = [X.shape[1]]
        
        self.weights_init = weights_init
        self.rate = rate
        self.steps = steps
        
        self.weights_trained = None
        self.train()

        
    
    
    def train(self, weights_init_new = None):
        # Use new weights if provided, otherwise use existing or generate new
        if weights_init_new is not None:
            self.weights_init = weights_init_new
        if self.weights_init is None:
            self.weights_init = []
            layers_size = [self.X.shape[1]] + self.internal_layers + [self.Y.shape[1]]
            for i in range(len(layers_size) - 1):
                self.weights_init.append(
                    np.random.randn(layers_size[i + 1], layers_size[i]) * np.sqrt(2 / layers_size[i])
                )
        
        self.weights_trained = nngrad.gradient_descent(
            self.weights_init, self.rate, self.steps, self.X, self.Y, self.activation_function
        )
        
        
    def feedforward_trained(self, x):
        if self.weights_trained is None:
            raise ValueError("Model has not been trained yet.")
        a, _ = nnback.feedforward(x, self.weights_trained, self.activation_function)
        return a[-1]
    
    
    def outcome(self, x):
        feedforward = self.feedforward_trained(x)
        return np.eye(feedforward.shape[1], dtype=int)[np.argmax(feedforward, axis=1)]
        
        
    
      