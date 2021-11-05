# -*- coding: utf-8 -*-
"""
Python script for testing scripts & functions
created on 03.11.2021 by jchburmester

"""
import numpy as np
from functions import sigmoid
from mlp import MLP

inputs = [1,0]
labels = 1
weights = np.random.normal(size=len(inputs)+1)


node_output = sigmoid(weights @ np.append(1, inputs))

new_instance = MLP(inputs)

output = new_instance.forward_step()

counter = 0

if np.abs(labels - output) >= 0.5:
    counter += 1
    
print(counter)

# next steps:
# store counter for accuracy calculation later
# compute error
# pass error backwards
# update weights
# training
