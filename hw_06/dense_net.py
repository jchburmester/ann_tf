import tensorflow as tf
from dense_block import DenseBlock
from transition_layer import TransitionLayer

class DenseNet(tf.keras.Model):
  """
  Custom DenseNet model, which inherits from the keras.Model class

  Architecture (default):

  3x3 Convolutional Layer
  Batch Normalization
  ReLu Activation
  Max Pool Layer
  6 Dense Block Layers
  Transition Layer
  12 Dense Block Layers
  Transition Layer
  24 Dense Block Layers
  Transition Layer
  16 Dense Block Layers
  Batch Normalization
  ReLu Activation
  Global Average Pool Layer
  Dense Layer

  Attributes:
  ----------
    growth_rate : int

    list_dense_in_block : list
        number of single dense blocks for each dense block
    n_dense_trans : int
        number of dense blocks / transition layer alternations
    layer_list : list
        list of all layers

  Functions:
  ---------
    call(inputs, training) : forward propagation
    train_step(input, target, loss_function, optimizer) : performs one train step
    test(data, loss_function) : evaluates model
  """

  def __init__(self, list_dense_in_block=[6,12,24,16], growth_rate=4):
    """
    Constructor

    Parameters:
    ----------
      list_dense_in_block : list
        stores the number of (single) dense blocks for each (big) dense block
      growth_rate : int

    """
    super(DenseNet, self).__init__()

    self.growth_rate = growth_rate
    self.list_dense_in_block = list_dense_in_block

    # number of dense blocks / transition layer alternations is length of passed list
    self.n_dense_trans = len(list_dense_in_block)

    # list to store all layers
    self.layer_list = []

    # initial convolutional layer
    # as we are not dealing with high resolution images, we don't need a stem that reduces the feature map size first
    self.layer_list.append(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    self.layer_list.append(tf.keras.layers.BatchNormalization())
    self.layer_list.append(tf.keras.layers.Activation(tf.nn.relu))
    self.layer_list.append(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

    # alternate dense and transition blocks
    for i in range(self.n_dense_trans):
      if (i!=(self.n_dense_trans-1)):
        # add as many (single) dense blocks as specified in list
        for _ in range(self.list_dense_in_block[i]):
          self.layer_list.append(DenseBlock())
        # only one transition layer after a block of dense blocks
        # this layer should reduce number of feature maps to number of conv layers in dense blocks * growth rate
        # the number of conv layers is number of dense blocks * 2 (because there are 2 conv layers in each dense block)
        self.layer_list.append(TransitionLayer((list_dense_in_block[i]*2) * self.growth_rate))
      # don't end with transition layer
      else:
        for _ in range(self.list_dense_in_block[i]):
          self.layer_list.append(DenseBlock())

    # add batch norm, global average pooling and output layer
    self.layer_list.append(tf.keras.layers.BatchNormalization())
    self.layer_list.append(tf.keras.layers.Activation(tf.nn.relu))
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
