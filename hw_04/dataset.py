import pandas as pd
import numpy as np
import tensorflow as tf

# load the data
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';', delimiter=None)

# inspect the dataset
print("The keys of the wine_quality dataset are:", *list(wine.columns), sep='\n')

# convert to tensorflow dataset
wine_slices = tf.data.Dataset.from_tensor_slices(wine)

"""
# split dataset
training_data = data.sample(frac=0.7, random_state=1)
val_data = data.sample(frac=0.15, random_state=1)
test_data = data.sample(frac=0.15, random_state=1)

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
ds = tf.data.Dataset.from_tensor_slices(tensors)
"""