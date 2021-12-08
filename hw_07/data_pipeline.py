def prepare_signal_data(signal_ds, batch_size):
  """
  Data pipeline for costum signal noise dataset:
  Shuffle, batch, prefetch

  Parameters:
  ----------
    signal_ds : tf dataset
      dataset to preprocess
    batch_size : int
      which batch size to use

  Returns:
  -------
    the resulting preprocessed dataset
  """
  # shuffle, batch, prefetch
  signal_ds = signal_ds.shuffle(1000)
  signal_ds = signal_ds.batch(batch_size)
  signal_ds = signal_ds.prefetch(20)

  return signal_ds
