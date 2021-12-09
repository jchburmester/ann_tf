import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_data import create_and_split_signal_dataset
from data_pipeline import prepare_signal_data
from lstm import MyLSTM

train_ds, val_ds, test_ds = create_and_split_signal_dataset()

# set the batch size used in data preprocessing
batch_size = 32

# preprocess the training, validation and test examples
train_preprocessed = prepare_signal_data(train_ds, batch_size)
val_preprocessed = prepare_signal_data(val_ds, batch_size)
test_preprocessed = prepare_signal_data(test_ds, batch_size)

## training
# for each batch, call model and get output

tf.keras.backend.clear_session()

# set the hyperparameters
num_epochs = 10
learning_rate = 0.001

# initialize model, loss and optimizer
LongShorts = MyLSTM()
cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# initialize lists for later visualization.
train_losses = []

test_losses = []
test_accuracies = []

# testing once before we begin
test_loss, test_accuracy = LongShorts.test(test_preprocessed, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on training data once before we begin
train_loss, _ = LongShorts.test(train_preprocessed, cross_entropy_loss)
train_losses.append(train_loss)

# train for specified number of epochs
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_preprocessed:
        train_loss = LongShorts.train_step(input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing, so we can track accuracy and test loss
    test_loss, test_accuracy = LongShorts.test(test_preprocessed, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# test on unseen validation dataset
val_loss, val_accuracy = LongShorts.test(val_preprocessed, cross_entropy_loss)

