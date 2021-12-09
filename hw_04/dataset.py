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

# separate labels from input
input_train = train.drop(columns='quality')
labels_train = train['quality']

input_val = validate.drop(columns='quality')
labels_val = validate['quality']

input_test = test.drop(columns='quality')
labels_test = test['quality']

# convert to tensors
pd_dataframes = [input_train, labels_train, input_val, labels_val, input_test, labels_test]

# convert to tensorflow datasets
tf_train = tf.data.Dataset.from_tensor_slices( (pd_dataframes[0], pd_dataframes[1]) )
tf_val = tf.data.Dataset.from_tensor_slices( (pd_dataframes[2], pd_dataframes[3]) )
tf_test = tf.data.Dataset.from_tensor_slices( (pd_dataframes[4],pd_dataframes[5]) )

# store in list and make datasets global
# for handing over to other scripts
global data
data = [tf_train, tf_val, tf_test]