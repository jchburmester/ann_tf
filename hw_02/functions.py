# -*- coding: utf-8 -*-
"""
Python script for helper functions
created on 03.11.2021 by jchburmester

"""

import numpy as np

# implementing sigmoid function & its derivative

def sigmoid(x):
    if isinstance(x, (int, float)):
       return 1/(1+np.exp(-x))
    else:
        result = [1/(1+np.exp(-i)) for i in x]
        return np.array(result)

def sigmoidprime(x):
    if isinstance(x, (int, float)):
       return sigmoid(x)*(1-sigmoid(x))
    else:
        result = [sigmoid(i)*(1-sigmoid(i)) for i in x]
        return np.array(result)
    
def classification(target, prediction, threshold=0.5):
    # Correct classification: target=1 and prediction>threshold OR target=0 and prediction<=threshold
    if target == 1:
        return prediction > threshold
    else:
        return prediction <= threshold