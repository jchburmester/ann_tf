import tensorflow as tf

def training_loop(model, train_preprocessed, val_preprocessed, test_preprocessed, num_epochs=10, learning_rate=0.001):
  """
  Costum training loop with Categorical Cross Entropy Loss and
  Adam Optimizer

  Parameters:
  ----------
    model : tf.keras.Model
      the model which we want to train
    num_epochs : int
      number of epochs
    learning_rate : float
      learning rate to use

  Returns:
  -------
    results (list of training, validation and test loss and accuracies)
  """

  tf.keras.backend.clear_session()

  # initialize loss and optimizer
  loss = tf.keras.losses.CategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam(learning_rate)

  # initialize lists for later visualization.
  train_losses = []

  val_losses = []
  val_accuracies = []

  # testing once before we begin
  val_loss, val_accuracy = model.test(val_preprocessed, loss)
  val_losses.append(val_loss)
  val_accuracies.append(val_accuracy)

  # check how model performs on training data once before we begin
  train_loss, _ = model.test(train_preprocessed, loss)
  train_losses.append(train_loss)

  # train for specified number of epochs
  for epoch in range(num_epochs):
      print(f'Epoch: {str(epoch)} starting with accuracy {val_accuracies[-1]}')

      # training (and checking in with training)
      epoch_loss_agg = []
      for input,target in train_preprocessed:
          train_loss = model.train_step(input, target, loss, optimizer)
          epoch_loss_agg.append(train_loss)

      # track training loss
      train_losses.append(tf.reduce_mean(epoch_loss_agg))

      # testing, so we can track val accuracy and val loss
      val_loss, val_accuracy = model.test(val_preprocessed, loss)
      val_losses.append(val_loss)
      val_accuracies.append(val_accuracy)

  # test on unseen test dataset
  test_loss, test_accuracy = model.test(test_preprocessed, loss)

  results = [train_losses, val_losses, val_accuracies, test_loss, test_accuracy]

  return results