import tensorflow as tf
from residual_block import ResidualBlock


class ResNet(tf.keras.Model):
  """
  Custom ResNet model, which inherits from the keras.Model class

  Architecture (default):

  3x3 Convolutional Layer
  20 Residual Block Layers
  Global Average Pool Layer
  Dense Layer

  Attributes:
  ----------
    res_block_number : int
        number of residual blocks
    layer_list : list
        list of all layers

  Functions:
  ---------
    call(inputs, training) : forward propagation
    train_step(input, target, loss_function, optimizer) : performs one train step
    test(data, loss_function) : evaluates model
  """

  def __init__(self, res_block_number=20):
    """
    Constructor

    Parameters:
    ----------
      res_block_number : int
        number of residual blocks
    """
    super(ResNet, self).__init__()

    self.res_block_number = res_block_number

    # list to store all layers
    self.layer_list = []

    # add convolutional layer
    self.layer_list.append(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

    # add residual blocks
    for i in range(self.res_block_number):
      self.layer_list.append(ResidualBlock())

    # add global pooling and output layer
    self.layer_list.append(tf.keras.layers.GlobalAveragePooling2D())
    self.layer_list.append(tf.keras.layers.Dense(10, activation='softmax'))


  @tf.function
  def call(self, inputs, training=True):
    """
    Computes the output of the model

    Parameters:
    ----------
      inputs : tensor
        input of the model
      training : bool
        indicates wheter we are in training mode
    Returns:
    -------
      the networks output
    """
    # feed input into first layer
    x = self.layer_list[0](inputs, training=training)

    # iterate through the remaining layers
    for layer in self.layer_list[1:]:
      x = layer(x, training=training)

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
