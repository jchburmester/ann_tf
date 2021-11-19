import pandas as pd
import numpy as np
import tensorflow as tf

# load the data
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';', delimiter=None)

# inspect the dataset
#print("The keys of the wine_quality dataset are:", *list(wine.columns), sep='\n')

# get median of wine quality column
global median
median = np.median(wine['quality'])

# split dataset
train, validate, test = np.split(wine.sample(frac=1, random_state=42), [int(.7*len(wine)), int(.85*len(wine))])

# split dataset
#training_data = wine.sample(frac=0.7, random_state=1)
#val_data = wine.sample(frac=0.15, random_state=1)
#test_data = wine.sample(frac=0.15, random_state=1)

# separate labels from input
input_train = train.drop(columns='quality')
labels_train = train['quality']

input_val = validate.drop(columns='quality')
labels_val = validate['quality']

input_test = test.drop(columns='quality')
labels_test = test['quality']

# convert to tensors
pd_dataframes = [input_train, labels_train, input_val, labels_val, input_test, labels_test]
tensors = [tf.convert_to_tensor(d) for d in pd_dataframes]

# convert to tensorflow datasets
tf_train = tf.data.Dataset.from_tensor_slices( (tensors[0], tensors[1]) )
tf_val = tf.data.Dataset.from_tensor_slices( (tensors[2], tensors[3]) )
tf_test = tf.data.Dataset.from_tensor_slices( (tensors[4],tensors[5]) )

# store in list and make datasets global
# for handing over to other scripts
global data
data = [tf_train, tf_val, tf_test]