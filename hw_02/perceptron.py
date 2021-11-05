# -*- coding: utf-8 -*-
"""
Python script for the implementation of the perceptron
created on 03.11.2021 by jchburmester

"""

import numpy as np
from functions import sigmoid

# class for creating, activating, and updating perceptron
class Perceptron:

    def __init__(self, input_units, alpha=1, act_func=sigmoid):
        """Initialise Perceptron"""
        self.input_units = input_units
        self.weights = np.random.normal(size=input_units+1)
        self.alpha = alpha
        self.act_func = act_func
        self.inputs = 0
        self.drive = 0
    
    def activate(self, inputs):
        """Activate Perceptron"""
        # add bias to inputs
        self.inputs = np.insert(inputs, 0, 1)
        
        # dot product of weights and inputs
        self.drive = self.weights @ self.inputs
        
        # activate node
        node_output = self.act_func(self.drive)
        
        return node_output
    
    def update(self, delta):
        """Update the weights and the bias with error term delta"""
        gradients = delta * self.inputs
        self.weights -= self.alpha * gradients
        
        return None