import pandas as pd
import numpy as np
import tensorflow as tf

# load the data
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';', delimiter=None)

# inspect the dataset
print("The keys of the wine_quality dataset are:", *list(wine.columns), sep='\n')

# split dataset
training_data = wine.sample(frac=0.7, random_state=1)
val_data = wine.sample(frac=0.15, random_state=1)
test_data = wine.sample(frac=0.15, random_state=1)

# separate labels from input
input_train = training_data.drop(columns='quality')
labels_train = training_data['quality']

input_val = val_data.drop(columns='quality')
labels_val = val_data['quality']

input_test = test_data.drop(columns='quality')
labels_test = test_data['quality']

# convert to tensors
pd_dataframes = [input_train, labels_train, input_val, labels_val, input_test, labels_test]
tensors = [tf.convert_to_tensor(d) for d in pd_dataframes]

# convert to tensorflow datasets
tf_train = tf.data.Dataset.from_tensor_slices(tensors[0])
tf_train_labels = tf.data.Dataset.from_tensor_slices(tensors[1])

tf_val = tf.data.Dataset.from_tensor_slices(tensors[2])
tf_val_labels = tf.data.Dataset.from_tensor_slices(tensors[3])

tf_test = tf.data.Dataset.from_tensor_slices(tensors[4])
tf_test_labels = tf.data.Dataset.from_tensor_slices(tensors[5])

print(len(tf_train))