import numpy as np
import tensorflow as tf

def integration_task(seq_len, num_samples):
    """
    Generator Function that yields a random noise signal and a target
    for num_samples times.
    
    Parameters:
    ----------
    seq_len : int
        length of noise (number of time steps) 
    num_samples : int
        number of samples
    """
    
    for _ in range(num_samples):
        # random noise signal of size seq_length
        signal = np.random.normal(size=seq_len)
        # target 1: integral of noise signal is larger/equal than 1
        # target 0: integral of noise signal is smaller than 1
        if(np.sum(signal,axis=-1) >= 1):
            target = 1
        else:
            target = 0

        # output shape of signal should be (seq_len,1)
        signal = np.expand_dims(signal,axis=-1)
        # output shape of target should be (1)
        target = np.expand_dims(target,axis=-1)
    
        yield (signal, target)


def my_integration_task():
    """
    Wrapper generator
    """
    # specify seq_len and num_samples
    my_integration_task = integration_task(5,80000)

    # interate through integration_task and yield function's yield
    for i in my_integration_task:
      yield i


def dataset_split(ds, ds_size, train_prop=0.7, val_prop=0.15, test_prop=0.15):
  """
  Costum dataset split into training, validation and test data

  Parameters:
  ----------
    ds : tensorflow dataset
      dataset to split
    ds_size : int
      size of the dataset
    train_prop, val_prop, test_prop : float
      split proportions

  Returns:
  -------
    the resulting train, validation and test datasets
  """

  # proportions must add up to 1
  if(train_prop + val_prop + test_prop != 1):
    return print("split sizes must sum up to 1")

  train_size = int(train_prop * ds_size)
  val_size = int(val_prop * ds_size)

  # take the respective numbers of examples, make sure the sets don't overlap
  train_ds = ds.take(train_size)
  val_ds = ds.skip(train_size).take(val_size)
  test_ds = ds.skip(train_size).skip(val_size)

  return train_ds, val_ds, test_ds


def create_and_split_signal_dataset():
    """
    Creates Tensorflow random noise signal dataset with from_generator.
    Splits ds into training, validation and test.

    Returns:
    ------
        splitted training, validation and test tf datasets
    """
    # create tf dataset
    signal_ds = tf.data.Dataset.from_generator(my_integration_task, output_signature=(tf.TensorSpec(shape=(10,1),
    dtype=tf.float32), tf.TensorSpec(shape=(1,),dtype=tf.float32)))

    # get dataset size
    ds_size = sum(1 for _ in signal_ds)

    train_ds, val_ds, test_ds = dataset_split(signal_ds, ds_size)

    return train_ds, val_ds, test_ds