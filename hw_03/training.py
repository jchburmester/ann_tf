# -*- coding: utf-8 -*-
"""
Python script for training the model
created on 03.11.2021

"""

import tensorflow as tf 
import numpy as np
from model import Model
from dataset import ds_test, ds_train

tf.keras.backend.clear_session()

model = Model()

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

test_loss, test_accuracy = model.test(ds_test, loss)
accuracies = []
losses = []
accuracies.append(test_accuracy)
losses.append(test_loss)

print('\n','Initial loss:', test_loss.numpy(), 'Initial accuracy:', test_accuracy.numpy(),'\n')

for epoch in range(10):
    print(f'Epoch: {epoch}, accuracy of {accuracies[-1]}')

    epoch_loss = []

    for (input, target) in ds_train:
        loss_value = model.training(input, target, loss, optimizer)
        epoch_loss.append(loss_value)

    epoch_loss.append(tf.reduce_mean(epoch_loss))

    test_loss, test_accuracy = model.test(ds_test, loss)
    accuracies.append(test_accuracy)
    losses.append(test_loss)