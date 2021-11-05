# -*- coding: utf-8 -*-
"""
Python script for the implementation of the MLP
created on 03.11.2021 by jchburmester

"""
import numpy as np
from perceptron import Perceptron
from functions import sigmoidprime

class MLP:
    
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.hidden_layer = [Perceptron(2), Perceptron(2), Perceptron(2), Perceptron(2)]
        self.output_layer = Perceptron(4)
        self.final_input = 0
        self.output = 0
        
    def forward_step(self):
        # inputs are passed through the network
        hidden_layer_outputs = []
        
        # compute outputs for each node
        for node in self.hidden_layer:
            # store outputs in a list
            hidden_layer_outputs.append(node.activate(self.inputs))
            
        # store input of final neuron to use in backward step
        self.final_input = hidden_layer_outputs
        
        # take outputs of hidden layer and activate final node with it
        self.output = self.output_layer.activate(np.array(hidden_layer_outputs))
                
        return self.output
        
        
    def backward_step(self):
        # first, compute error
        error = self.labels - self.output
        
        # compute delta
        delta_output_neuron = - error * sigmoidprime(self.final_input)
        
       # compute deltas for each node
        for idx, node in enumerate(self.hidden_layer):
            # slice layer to ignore bias
            output_layer_weights_no_bias = self.output_layer.weights[1:]
            # deltas
            delta = sigmoidprime(node.drive) * delta_output_neuron * output_layer_weights_no_bias[idx]
            # update node with delta
            node.update(delta)
        
        # update weights of final node
        # has to be in this order since we need to compute deltas of hidden layer before weights are updated
        self.output_layer.update(delta_output_neuron)
        
        return None
        
    
    
    
    
    
    
    
    