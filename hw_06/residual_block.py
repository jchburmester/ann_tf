import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
  """
  Instantiates the layers and computations involved in a residual block from ResNet

  Attributes:
  ----------
    n_filters : int
      number of filters used by the first convolutions
    out_filters : int
      number of channels of the output
    input_resize : tf.keras.layers.Layer
        layer for shaping input tensor so that dimensions match for adding
    layer_list : list
        list of all layers, dependend on mode used

  Functions:
  ---------
    call(inputs) : forward propagation
    """

  def __init__(self, n_filters=32, out_filters=256):
    """
    Constructor

    Parameters:
    ----------
      n_filters : int
        number of filters used by the first convolutions
      out_filters : int
        number of channels of the output
    """
    super(ResidualBlock, self).__init__()

    self.n_filters = n_filters
    self.out_filters = out_filters

    # resize layer
    self.input_resize = tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size=(1,1))

    # use batch normalization and a non-linearity (relu)
    self.layer_list = [tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation(tf.nn.relu), tf.keras.layers.Conv2D(filters=self.n_filters, kernel_size =(1,1)),
                      tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation(tf.nn.relu), tf.keras.layers.Conv2D(filters=self.n_filters, kernel_size =(3,3), padding="same"),
                      tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation(tf.nn.relu), tf.keras.layers.Conv2D(filters=self.out_filters, kernel_size =(1,1))
                      ]

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

    # transform original input to also have out_filters channels
    x_shaped = self.input_resize(inputs)

    # Add x_shaped and x_out
    x_out = tf.keras.layers.Add()([x_out, x_shaped])

    return x_out
