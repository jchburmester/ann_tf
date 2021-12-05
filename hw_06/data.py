import tensorflow as tf
import tensorflow_datasets as tfds

def load_cifar():
    """
    Load the Cifar10 dataset from tensorflow datasets.
    Print info and show example images.
    Concatenate train and test data to manually split later.

    Returns:
    -------
        the complete cifar10 dataset
        size of the dataset
    """
    (train_ds, test_ds), ds_info = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

    # dataset size is sum of train and test examples
    ds_size = ds_info.splits['train'].num_examples + ds_info.splits['test'].num_examples

    # show info about the dataset
    print(ds_info)

    # show some examples
    tfds.show_examples(train_ds, ds_info)

    cifar_ds = train_ds.concatenate(test_ds)

    return cifar_ds, ds_size


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



def prepare_cifar_data(cifar, batch_size):
  """
  Data pipeline for cifar10 dataset:
  Input normalization, one-hot targets, shuffle, batch, prefetch

  Parameters:
  ----------
    cifar : dataset
      dataset to preprocess
    batch_size : int
      which batch size to use

  Returns:
  -------
    the resulting preprocessed dataset
  """
  # convert data from uint8 to float32
  cifar = cifar.map(lambda img, target: (tf.cast(img, tf.float32), target))
  # input normalization
  cifar = cifar.map(lambda img, target: ((img/128.)-1., target))
  # create one-hot targets
  cifar = cifar.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  # cache this progress in memory
  cifar = cifar.cache()
  # shuffle, batch, prefetch
  cifar = cifar.shuffle(1000)
  cifar = cifar.batch(batch_size)
  cifar = cifar.prefetch(20)

  return cifar
