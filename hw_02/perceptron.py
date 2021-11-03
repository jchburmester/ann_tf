# -*- coding: utf-8 -*-
"""
Python script for the implementation of the perceptron
created on 03.11.2021 by jchburmester

"""

import numpy as np
from functions import sigmoid

# class for creating, activating, and updating perceptron
class Perceptron:

    def __init__(self, input_units, alpha=1):
        self.input_units = input_units
        self.weights = np.random.normal(size=input_units+1)
        self.alpha = alpha
        self.inputs = 0
    
    def forward_step(self, inputs):
        return self.act_func()