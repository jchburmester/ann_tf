import tensorflow as tf
from load_data import create_and_split_signal_dataset
from data_pipeline import prepare_signal_data
from lstm import My_LSTM
from visualization import visualize
from training import training_loop


train_ds, val_ds, test_ds = create_and_split_signal_dataset()

# set the batch size used in data preprocessing
# ds sizes must be aliquot by batch_size
batch_size = 50

# preprocess the training, validation and test examples
train_preprocessed = prepare_signal_data(train_ds, batch_size)
val_preprocessed = prepare_signal_data(val_ds, batch_size)
test_preprocessed = prepare_signal_data(test_ds, batch_size)

# create LSTM model, train, show summary and visualize results
lstm_model = My_LSTM(batch_size)
with tf.device('/device:gpu:0'):
    results_simple = training_loop(lstm_model, train_preprocessed, val_preprocessed, test_preprocessed)
visualize(results_simple)