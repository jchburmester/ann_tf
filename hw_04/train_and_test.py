import tensorflow as tf
from dataset import ds_list
from model import Wine_Tasting

# unravel data
ds_train = ds_list[0]


# training
# 10 epochs
# learning rate: 0.1
# loss: binary cross entropy
# optimiser: SGD

# clearing backend
tf.keras.backend.clear_session()

""" Initialising an instance of the model """
model = Wine_Tasting()

""" Initialising the loss and optimiser function """
loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# some empty lists
accuracies = []
losses = []
train_losses = []
train_accuracies = []

# 
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