import tensorflow as tf
import tensorflow_datasets as tfds
import data_preprocessing as pre
from simple_cnn import ConvModel
from res_net import ResNet
from dense_net import DenseNet
from training import training_loop
from visualization import visualize

# load dataset and get dataset size
cifar10, ds_size = pre.load_cifar()

# split dataset into train, val and test
train_ds, val_ds, test_ds = pre.dataset_split(cifar10, ds_size)

# set the batch size used in data preprocessing
batch_size = 64

# preprocess the training, validation and test examples
train_preprocessed = pre.prepare_cifar_data(train_ds, batch_size)
val_preprocessed = pre.prepare_cifar_data(val_ds, batch_size)
test_preprocessed = pre.prepare_cifar_data(test_ds, batch_size)


# create ResNet model, train, show summary and visualize results
res_net_model = ResNet()
with tf.device('/device:gpu:0'):
    results_res_net = training_loop(res_net_model)
res_net_model.build((64,32,32,3))
res_net_model.summary()
visualize(results_res_net)

# create DenseNet model, train, show summary and visualize results
dense_net_model = DenseNet()
with tf.device('/device:gpu:0'):
    results_dense_net = training_loop(dense_net_model)
dense_net_model.build((64,32,32,3))
dense_net_model.summary()
visualize(results_dense_net)

# create simple CNN model, train, show summary and visualize results
simple_model = ConvModel()
with tf.device('/device:gpu:0'):
    results_simple = training_loop(simple_model)
simple_model.build((64,32,32,3))
simple_model.summary()
visualize(results_simple)
