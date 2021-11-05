# -*- coding: utf-8 -*-
"""
Python script for the implementation of the perceptron
created on 03.11.2021 by jchburmester

"""

import numpy as np
from functions import sigmoid
from functions import sigmoidprime

# class for creating, activating, and updating perceptron
class Perceptron:

    def __init__(self, input_units, alpha=1, act_func=sigmoid):
        """Initialise Perceptron"""
        self.input_units = input_units
        self.weights = np.random.normal(size=input_units+1)
        self.alpha = alpha
        self.act_func = act_func
        self.input = 0
    
    def activate(self, inputs):
        """Activate Perceptron"""
        self.input = self.weights @ np.append(1, inputs)
        node_output = self.act_func(self.input)
        
        return node_output
    
    def update(self, delta):
        """Update the weights and the bias with error term delta"""
           
        
        return None