import tensorflow as tf


class TransitionLayer(tf.keras.layers.Layer):
  """
  Costum transition network part. Reduces size of feature maps and number of feature maps

  Attributes:
  ----------
    reduce_filters_to : int
        desired (reduced) number of feature maps
    layer_list : list
        list of all layers

  Functions:
  ---------
    call(inputs, training) : forward propagation
  """

  def __init__(self, n_filters):
    """
    Constructor

    Parameters:
    ----------
      n_filters : int
        desired (reduced) number of feature maps
    """
    super(TransitionLayer, self).__init__()

    # number of feature maps should be reduced
    self.reduce_filters_to = n_filters

    # conv layer reduces number of feature maps, pooling layer reduces size of feature maps
    self.layer_list = [tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation(tf.nn.relu), tf.keras.layers.Conv2D(self.reduce_filters_to, kernel_size=(1,1), padding="valid", use_bias=False),
                       tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="valid")]

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
