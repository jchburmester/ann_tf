import tensorflow as tf
from tensorflow.keras.layers import Dense

class Wine_Tasting(tf.keras.Model):

    def __init__(self):
        """ initialising the model """
        super(Wine_Tasting, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.sigmoid)
        self.ciao = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    @tf.function
    def call(self, inputs):
        """ forward step """
        signal = self.dense1(inputs)
        signal = self.dense2(signal)
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

            # np.round(,0) instead of np.argmax?
            sample_accuracy = np.argmax(pred, axis=1) == np.argmax(target, axis=1)
            sample_accuracy = np.mean(sample_accuracy)

            test_accuracy.append(np.mean(sample_accuracy))
            test_loss.append(loss_value.numpy())

        test_loss = tf.reduce_mean(test_loss)
        test_accuracy = tf.reduce_mean(test_accuracy)

        return test_loss, test_accuracy