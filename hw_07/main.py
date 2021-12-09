import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_data import create_and_split_signal_dataset
from data_pipeline import prepare_signal_data
from lstm import MyLSTM


train_ds, val_ds, test_ds = create_and_split_signal_dataset()

# set the batch size used in data preprocessing
batch_size = 64

# preprocess the training, validation and test examples
train_preprocessed = prepare_signal_data(train_ds, batch_size)
val_preprocessed = prepare_signal_data(val_ds, batch_size)
test_preprocessed = prepare_signal_data(test_ds, batch_size)



