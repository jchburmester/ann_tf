# -*- coding: utf-8 -*-
"""
Python script for training the MLP
created on 03.11.2021 by jchburmester

"""

import numpy as np
import itertools
from mlp import MLP

# creating inputs
inputs = np.array(list(itertools.product([False, True], repeat=2)))

# creating labels for logical gates
label_and = np.array([False, False, False, True]) # logical 'and'
label_or = np.array([False, True, True, True]) # logical 'or'
label_not_and = np.array([True, True, True, False]) # logical 'not and'
label_not_or = np.array([True, False, False, False]) # logical 'not or'
label_xor = np.array([False, True, True, False]) # logical 'xor'


def training(epochs=1000):
    for epoch in epochs:
        


## training
# create instance of MLP
# train instance for 1000 epochs
# in each epoch, loop over each point in dataset once
# for each data point, perform forward and backward step
# record accuracy, and loss for each point


# loss function (in functions?)
# squared error np.sqrt(t - y)
# accuracy: if loss smaller than 0.5, classified positive
# if larger than 0.5, classified negative
# overall accuracy: ratio of correct classifications vs. total classifications