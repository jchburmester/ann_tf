import tensorflow as tf
from dataset import data
from model import Wine_Tasting
from pipeline import doItForTheWine

# unravel data
ds_train = data[0]
ds_val = data[1]
ds_test = data[2]

# put data through pre-processing pipeline
ds_train = doItForTheWine(ds_train)
ds_val = doItForTheWine(ds_val)
ds_test = doItForTheWine(ds_test)

# training
# 10 epochs
# learning rate: 0.1
# loss: binary cross entropy
# optimiser: SGD

def training_loop(my_optimizer):

    # clearing backend
    tf.keras.backend.clear_session()

    """ Initialising an instance of the model """
    model = Wine_Tasting()

    """ Initialising the loss and optimiser function """
    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = my_optimizer(learning_rate=0.1)

    # some empty lists
    global accuracies
    global losses
    global train_losses

    accuracies = []
    losses = []
    train_losses = []
    train_accuracies = []

    test_loss, test_accuracy = model.test(ds_test, loss)

    accuracies.append(test_accuracy)
    losses.append(test_loss)

    train_loss, train_accuracy = model.test(ds_train, loss)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print('Initial loss:', test_loss.numpy(), 'Initial accuracy:', test_accuracy.numpy(),'\n')

    """ Train the model for 10 epochs """
    for epoch in range(10):
        print(f'Epoch: {epoch}, accuracy of {accuracies[-1]}')

        epoch_loss = []

        for (input, target) in ds_train:
            loss_value = model.training(input, target, loss, optimizer)
            epoch_loss.append(loss_value)

        train_losses.append(tf.reduce_mean(epoch_loss))

        test_loss, test_accuracy = model.test(ds_test, loss)
        accuracies.append(test_accuracy)
        losses.append(test_loss)

training_loop(my_optimizer=tf.keras.optimizers.SGD)
#training_loop(my_optimizer=tf.keras.optimizers.Adam)
