import tensorflow as tf
import numpy as np

class LSTM_cell:

    def __init__(self, units):
        
        # weight init (kernel init)
        self.units = units
        self.forget_gate = tf.keras.layers.Dense(units, activation='sigmoid',
                                                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                                                bias_initializer=tf.keras.initializers.ones())
        self.input_gate = tf.keras.layers.Dense(units, activation='sigmoid')
        self.cell_state_candidates = tf.keras.layers.Dense(units, activation='tanh')
        self.output_gate = tf.keras.layers.Dense(units, activation='sigmoid')

    def call(self, x, states):
        
        ht_old = states[0]
        ct_old = states[1]

        # concat hidden state and input
        concat_input = tf.concat(x, ht_old, axis=0)

        # compute forget gate
        forget_output = self.forget_gate(concat_input)

        # compute input gate
        input = self.input_gate(concat_input)

        # compute cell state candidates
        tanh = self.cell_state_candidates(concat_input)

        # compute output gate
        output = self.output_gate(concat_input)

        ct_new = np.dot(ct_old, forget_output) + np.dot(input, tanh)
        ht_new = np.dot(tf.keras.activations.tanh(ct_new), output)

        return (ht_new, ct_new)

class LSTM_layer:
    def __init__(self, cell):

        super(LSTM_layer, self).__init__()
        self.cell = cell

    def call(self, x, states):
        
        # empty list for storing hidden layer output for every time step in sequence
        hidden_states = []

        # for each input, call LSTM cell and compute new states
        for i in range(x.shape[1]):
            states = self.cell.call(x[i], states)
            hidden_states.append(states[0])

        return hidden_states
        
    def zero_states(self, batch_size):
        self.states = tf.zeros((batch_size, 2))


class MyLSTM(tf.keras.Model):
    def __init__(self):
        super(MyLSTM, self).__init__()

        # read_in layer(s)
        self.embedding = tf.keras.layers.Dense(24, activation='sigmoid')

        # LSTM implementation: LSTM layer with a single LSTM cell
        self.LSTM = LSTM_layer(LSTM_cell(128))

        # final output layer to transform last output into prediction
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    @tf.function
    def call(self, x):
        # set zero states since for every batch we are calling the model again
        LSTM = self.LSTM.zero_states()

        read_in = self.embedding(x)
        outputs = LSTM.call(read_in)

        return self.out(outputs[-1])

    def train_step(self, signal, target, loss_function, optimizer):
        # use context manager
        with tf.GradientTape() as tape:
            prediction = self.call(signal)
            loss = loss_function(target, prediction)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def test(self, test_data, loss_function):
        # test over complete test data
        test_accuracy_aggregator = []
        test_loss_aggregator = []

        for (signal, target) in test_data:
            prediction = self.call(signal)
            sample_test_loss = loss_function(target, prediction)
            sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            sample_test_accuracy = np.mean(sample_test_accuracy)
            test_loss_aggregator.append(sample_test_loss.numpy())
            test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

        test_loss = tf.reduce_mean(test_loss_aggregator)
        test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

        return test_loss, test_accuracy
