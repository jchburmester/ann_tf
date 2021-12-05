import tensorflow as tf


class ConvModel(tf.keras.Model):
  """
  Custom model, which inherits from the keras.Model class

  Architecture:

  5x5 Convolutional Layer, 96 filter
  3x3 Convolutional Layer, 96 filter
  Max Pooling Layer
  3x3 Convolutional Layer, 96 filter
  3x3 Convolutional Layer, 96 filter
  Global Average Pooling Layer
  Dense Layer, 256 units
  Output Dense Layer

  Attributes:
  ----------
    layer_list : list
        list of all layers

  Functions:
  ---------
    call(inputs) : forward propagation
    train_step(input, target, loss_function, optimizer) : performs one training step
    test(test_data, loss_function) : tests the model
  """

  def __init__(self):
    """
    Constructor
    """
    super(ConvModel, self).__init__()

    # store layers in a list for iteration
    self.layer_list = [tf.keras.layers.Conv2D(filters=96, kernel_size=5, padding='valid', activation='relu'),
                       tf.keras.layers.Conv2D(filters=96, kernel_size=3, padding='valid', activation='relu'),
                       tf.keras.layers.MaxPool2D(2, strides=1),
                       tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu'),
                       tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid', activation='relu'),
                       tf.keras.layers.GlobalAveragePooling2D(),
                       tf.keras.layers.Dense(256, activation='relu'),
                       tf.keras.layers.Dense(10, activation='softmax')
                       ]

  @tf.function
  def call(self, inputs):
    """
    Computes the output of the model

    Parameters:
    ----------
      inputs : tensor
        input of the model

    Returns:
    -------
      the networks output
    """
    # feed input into first layer
    x = self.layer_list[0](inputs)

    # iterate through the remaining layers
    for layer in self.layer_list[1:]:
      x = layer(x)
    return x

  def train_step(self, input, target, loss_function, optimizer):
    """
    Performs one training step

    Parameters:
    ----------
      input : tensor
        input of the model
      target : tensor
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
