# -*- coding: utf-8 -*-
"""
Python script for the implementation of the MLP
created on 03.11.2021 by jchburmester

"""
import numpy as np
from perceptron import Perceptron
from functions import sigmoidprime

class MLP:
    
    def __init__(self, inputs):
        self.inputs = inputs
        self.hidden_layer = [Perceptron(2), Perceptron(2), Perceptron(2), Perceptron(2)]
        self.output_layer = Perceptron(4)
        self.output = 0
        
    def forward_step(self):
        # inputs are passed through the network
        hidden_layer_outputs = []
        
        # compute outputs for each node
        for node in self.hidden_layer:
            # store outputs in a list
            hidden_layer_outputs.append(node.activate(self.inputs))
        
        print(hidden_layer_outputs)
        # take outputs of hidden layer and activate final node with it
        final_node_output = self.output_layer.activate(np.array(hidden_layer_outputs))
        
        return final_node_output
        
        
    def backward_step(self):
        # parameters of the network are updated
        return None
        
        