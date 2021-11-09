# -*- coding: utf-8 -*-
"""
Python script for training the MLP
created on 03.11.2021 by jchburmester

"""

import numpy as np
import itertools
from mlp import MLP
from functions import classification

#  declaration of global variables
global epoch_training
global epoch_loss

# creating inputs
inputs = np.array(list(itertools.product([False, True], repeat=2)))

# creating labels for logical gates
label_and = np.array([False, False, False, True]) # logical 'and'
label_or = np.array([False, True, True, True]) # logical 'or'
label_not_and = np.array([True, True, True, False]) # logical 'not and'
label_not_or = np.array([True, False, False, False]) # logical 'not or'
label_xor = np.array([False, True, True, False]) # logical 'xor'


# training of the MLP network on a given logical gate
# returns loss & accuracy
def training(log_gate, epochs=1000):
    epochs = epochs
    
    # store loss & accuracy for each epoch
    epoch_losses = []
    epoch_accuracies = []
    
    # store loss & accuracy for each training step
    train_step_losses = [] 
    train_step_accuracies = []
    label = log_gate

    my_mlp = MLP(label)
    
    for epoch in range(epochs):
        # in each epoch, loop through each data pair and record loss & nr of correct classifications
        for idx in range(4):    
            y_hat = my_mlp.forward_step(inputs[idx])
            loss = (label[idx] - y_hat)**2
            train_step_losses.append(loss)
            my_mlp.backward_step(label[idx])
            # correct_classification function returns bool, we want integer
            train_step_accuracies.append(int(classification(label[idx],y_hat)))
    
        # epoch loss & accuracy are the means of the training steps
        epoch_losses.append(np.mean(train_step_losses))
        epoch_accuracies.append(np.mean(train_step_accuracies))
        
    return epoch_losses, epoch_accuracies

training_result = training(label_or)
epoch_loss = training_result[0]
epoch_accuracy = training_result[1]