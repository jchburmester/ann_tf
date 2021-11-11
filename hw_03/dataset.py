# -*- coding: utf-8 -*-
"""
Python script for pre-processing the pipeline
created on 03.11.2021

"""
import tensorflow as tf
import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
ds, ds_info = tfds.load('genomics_ood', shuffle_files=True, as_supervised=True, with_info=True)

print(ds['test'])

# .map(lambda x, y: (tf.cast(tf.reshape(y, (-1,)), y))
# .map(lambda x, y: ((x/255)-1, y))
# .map(lambda x, y: (x, tf.one_hot(y, depth=4)))
# .cache()
# training.pick(100000)
# test.pick(1000)
# .shuffle().batch().prefetch()