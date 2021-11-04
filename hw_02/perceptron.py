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
        
        self.input_units = input_units
        self.weights = np.random.normal(size=input_units+1)
        self.alpha = alpha
        self.act_func = act_func
    
    def activate(self, inputs):
        
        # calculate activation of perceptron
        node_output = self.act_func(self.weights @ np.append(1, inputs))
        
        return node_output
    
    def update(self, delta):
        
        # to update the parameters
        # compute gradients for weights and bias from error term
        
        return None