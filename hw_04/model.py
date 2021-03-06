import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense

class Wine_Tasting(tf.keras.Model):

    def __init__(self):
        """ initialising the model """
        super(Wine_Tasting, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(8, activation=tf.nn.leaky_relu)
        #self.dense3 = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid)
        self.ciao = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    @tf.function
    def call(self, inputs):
        """ forward step """
        signal = self.dense1(inputs)
        signal = self.dense2(signal)
        #signal = self.dense3(signal)
        signal = self.ciao(signal)
        
        return signal

    def training(self, input, target, loss, optimizer):
        """ performing gradients and train variables"""
        with tf.GradientTape() as tape:
            pred = self.call(input)
            loss_value = loss(target, pred)

        grads = tape.gradient(loss_value, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_value

    def test(self, test_data, loss):
        """ Evaluating metrics """
        test_accuracy = []
        test_loss = []

        for (input, target) in test_data:
            pred = self.call(input)
            loss_value = loss(target, pred)

            sample_accuracy = np.round(pred, 0) == np.round(target, 0)
            sample_accuracy = np.mean(sample_accuracy)

            test_accuracy.append(np.mean(sample_accuracy))
            test_loss.append(loss_value.numpy())

        test_loss = tf.reduce_mean(test_loss)
        test_accuracy = tf.reduce_mean(test_accuracy)

        return test_loss, test_accuracy