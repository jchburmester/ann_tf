# -*- coding: utf-8 -*-
"""
Python script for implementing the model
created on 03.11.2021

"""

import numpy as np
import tensorflow as tf

class Dense(tf.keras.layers.Layer):

    def __init__(self, units, activation):
        super(Dense, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape): 
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs): 
        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        return x



class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = Dense(units=256, activation=tf.nn.sigmoid)
        self.dense2 = Dense(units=256, activation=tf.nn.sigmoid)
        self.out = Dense(units=10, activation=tf.nn.softmax)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        
        return x

    def training(self, input, target, loss, optimizer):

        with tf.GradientTape() as tape:
            pred = self.call(input)
            loss_value = loss(target, pred)

        grads = tape.gradient(loss_value, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_value

    def test(self, test_data, loss):

        test_accuracy = []
        test_loss = []

        for (input, target) in test_data:
            pred = self.call(input)
            loss_value = loss(target, pred)

            sample_accuracy = np.argmax(pred, axis=1) == np.argmax(target, axis=1)
            sample_accuracy = np.mean(sample_accuracy)

            test_accuracy.append(np.mean(sample_accuracy))
            test_loss.append(loss_value.numpy())

        test_loss = tf.reduce_mean(test_loss)
        test_accuracy = tf.reduce_mean(test_accuracy)

        return test_loss, test_accuracy