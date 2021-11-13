# -*- coding: utf-8 -*-
"""
Python script for pre-processing the pipeline
created on 03.11.2021

"""
import tensorflow as tf
import tensorflow_datasets as tfds

ds_train, ds_test = tfds.load('genomics_ood', as_supervised=True, split=['train[:10%]','test[:1%]'])

print(ds_train.__len__()) # 100000
print(ds_test.__len__())  # 1000

print(ds_train.element_spec)

def onehotify(tensor):
    vocab = {'A':'1', 'C': '2', 'G':'3', 'T':'0'} 
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key]) 
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8) 
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot

def preprocessing(data):

    data = data.map(lambda sequence, target: (onehotify(sequence), target))
    data = data.map(lambda sequence, target: (sequence, tf.one_hot(target, 10)))
    data = data.cache()
    data = data.shuffle(1000)
    data = data.batch(10)
    data = data.prefetch(10)
    return data

ds_train = ds_train.apply(preprocessing)
ds_test = ds_test.apply(preprocessing)