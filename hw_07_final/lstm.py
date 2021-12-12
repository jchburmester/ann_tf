import tensorflow as tf
import numpy as np

class LSTM_cell():
    """
    Costum LSTM cell
    
    Attributes:
    ----------
      units : int
        number of units for layers
      forget_gate : tf.keras.layers.Layer
        forget gate
      input_gate : tf.keras.layers.Layer
        input gate
      memory_gate : tf.keras.layers.Layer
        cell memory gate
      output_gate : tf.keras.layers.Layer
        output gate

    Functions:
    ---------
      get_units() : Getter function for units

    """
    def __init__(self, units):
        """
        Constructor
        
        Parameters:
        ----------
            units : int
              number of units for layers
        """
        self.units = units

        # forget gate
        self.forget_gate = tf.keras.layers.Dense(units, activation='sigmoid', bias_initializer=tf.keras.initializers.ones())
        # input gate
        self.input_gate = tf.keras.layers.Dense(units, activation='sigmoid')
        # cell-memory gate
        self.memory_gate = tf.keras.layers.Dense(units, activation='tanh')
        # output gate
        self.output_gate = tf.keras.layers.Dense(units, activation='sigmoid')

    def get_units(self):
        """
        Getter function for units
        """
       
        return self.units


    def call(self, x, states):
        """
        One forward step in LSTM cell
        
        Parameters:
        ----------
            x : tf.tensor
              input
            states : tuple
              hidden and cell state
        
        Returns:
        -------
            tuple of next hidden and cell state
        """
        # old hidden state
        hold = states[0]
        # old cell state
        cold = states[1]

        # combine input and old hidden state
        x_and_hidden = tf.concat([x, hold], axis=-1)

        # feed into input gate
        i = self.input_gate(x_and_hidden)
        # feed into forget gate
        f = self.forget_gate(x_and_hidden)

        # new cell info
        new_info = tf.math.multiply(self.memory_gate(x_and_hidden), i)
        # forget old cell state
        forget_c = tf.math.multiply(f, cold)
        # new cell state
        next_c = forget_c + new_info

        # feed into output gate
        o = self.output_gate(x_and_hidden)

        # new hidden state
        next_h = tf.math.multiply(o, tf.keras.activations.tanh(next_c))

        return (next_h, next_c)


class LSTM_layer(tf.keras.layers.Layer):
    """
    Costum LSTM layer class

    Attributes:
    ----------
      cell : LSTM_cell
        costum LSTM cell
    
    Functions:
    ---------
      call(x, states) : Unrolling of LSTM layer
      zero_states(batch_size) : Initialize hidden and cell state
    """

    def __init__(self, cell):
        """
        Constructor
        
        Parameters:
        ----------
            cell : LSTM_cell
              costum LSTM cell
        """
        super(LSTM_layer, self).__init__()
        
        self.cell = cell

    def call(self, x, states):
        """
        Unrolling of LSTM layer

        Parameters:
        ----------
            x : tf.tensor
              input
            states : tuple
              hidden and cell state
        """
        # list for storing hidden states (to be able to access the output after arbitrary time steps)
        output_list = []
        
        # propagate through time steps 
        for i in range(x.shape[1]):
            states = self.cell.call(x[:,i,:], states)
            output_list.append(states[0])

        return output_list

    def zero_states(self, batch_size):
        """
        Initializes hidden and cell state to zeros
        
        Parameters:
        ----------
            batch_size : int
              batch size of the data
        """
        # shape should be (batch_size, cell_units)
        hidden_init = tf.zeros((batch_size, self.cell.get_units()))
        cell_init = tf.zeros((batch_size, self.cell.get_units()))

        return (hidden_init, cell_init)


class My_LSTM(tf.keras.Model):
    """
    Custom LSTM model

    Attributes:
    ----------
      embedding : tf.keras.layers.Layer
        read-in layer
      LSTM : custom layer
        LSTM layer with single LSTM cell
      out : tf.keras.layers.Layer
        output layer

    Functions:
    ---------
      call(x) : forward propagation
      train_step(input, target, loss_function, optimizer) : performs one training step
      test(data, loss_function) : evaluates model

    """
    def __init__(self, batch_size):
        """
        Constructor

        Parameters:
        ----------
          batch_size : int
            batch size of the data
        """
        super(My_LSTM, self).__init__()

        # read-in layer
        self.embedding = tf.keras.layers.Dense(units=24, activation='sigmoid')
        # LSTM layer with single LSTM cell
        self.LSTM = LSTM_layer(LSTM_cell(128))
        # output layer
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

        # initialize hidden and cell state to zeros
        self.init_states = self.LSTM.zero_states(batch_size)

    @tf.function
    def call(self, x):
        """
        Forward propagation

        Parameters:
        ----------
          x : tf.tensor
            input
        """
        
        x = self.embedding(x)
        all_hiddens = self.LSTM.call(x, self.init_states)
        x_out = self.out(all_hiddens[-1])

        return x_out

    def train_step(self, input, target, loss_function, optimizer):
        """
        Performs one training step

        Parameters:
        ----------
        input : tf.tensor
            input of the model
        target : tf.tensor
            true target
        loss_function : tf.keras.losses.Loss
            loss function
        optimizer : tf.keras.optimizers.Optimizer
            optimizer to be used

        Returns:
        -------
        the respective loss
        """

        # use context manager
        with tf.GradientTape() as tape:
            prediction = self.call(input)
            loss = loss_function(target, prediction)
            gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    def test(self, data, loss_function):
        """
        Evaluates model with validation / test dataset

        Parameters:
        ----------
        data : tensorflow dataset
            the preprocessed validation / test dataset
        loss_function : tf.keras.losses.Loss
            loss function

        Returns:
        -------
        loss and accuracy
        """
        # evaluate over complete test data
        accuracy_aggregator = []
        loss_aggregator = []

        for (input, target) in data:
            prediction = self.call(input)
            sample_loss = loss_function(target, prediction)
            sample_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
            sample_accuracy = np.mean(sample_accuracy)
            loss_aggregator.append(sample_loss.numpy())
            accuracy_aggregator.append(np.mean(sample_accuracy))

        loss = tf.reduce_mean(loss_aggregator)
        accuracy = tf.reduce_mean(accuracy_aggregator)

        return loss, accuracy
