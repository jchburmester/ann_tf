import tensorflow as tf


class DenseBlock(tf.keras.layers.Layer):
  """
  Instantiates the layers and computations involved in a a DenseBlock from DenseNet

  Attributes:
  ----------
    layer_list : list
        list of all layers
  """

  def __init__(self, n_filters=128, new_channels=32):
    """
    Constructor

    Parameters:
    ----------
      n_filters : int
        number of filters used within the block (does not have an effect on n of output channels)
      new_channels : int
        number of channels to be added to the input by the block
    """
    super(DenseBlock, self).__init__()

    # block consists of several batchnorm and relu, one 1x1 conv layer and one 3x3 layer to be concatenated with the input
    self.layer_list = [tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation(tf.nn.relu), tf.keras.layers.Conv2D(n_filters, kernel_size=(1,1), padding="valid"),
                       tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation(tf.nn.relu), tf.keras.layers.Conv2D(new_channels, kernel_size=(3,3), padding="same", use_bias=False)]

  @tf.function
  def call(self, inputs, training=True):
    """
    Computes the output of the model

    Parameters:
    ----------
      inputs : tensor
        input of the model
      training : bool
        indicates whether we are in training or inference mode

    Returns:
    -------
      the networks output
    """
    # feed input into first layer
    # set training flag because first layer is batch normalization
    x_out = self.layer_list[0](inputs, training=training)

    # iterate through the remaining layers
    for layer in self.layer_list[1:]:
      # if batch normalization layer, set training flag
      if(type(layer) == tf.keras.layers.BatchNormalization):
        x_out = layer(x_out, training=training)
      else:
        x_out = layer(x_out)

    # Concatenate layer (tf.keras.layers.Layer that calls tf.concat)
    # axis -1 is the channel dimension
    x_out = tf.keras.layers.Concatenate(axis=-1)([inputs, x_out])

    return x_out
