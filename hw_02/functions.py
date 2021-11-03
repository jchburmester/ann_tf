# -*- coding: utf-8 -*-
"""
Python script for additional/helper functions
created on 03.11.2021 by jchburmester

"""

import numpy as np
import matplotlib as plt

# implementing sigmoid function & its derivative

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidprime(x):
    return sigmoid(x)*(1-sigmoid(x))