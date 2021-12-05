###To-Do:
- Set hyperparameters to increase performance (alter the number of residual blocks / the number of dense blocks)
- growth_rate should be the only param for DenseNet
- Understand and implement growth rate (--> DenseNet)
- Short explanation of findings
- upload to GitHub
- table for findings

# Homework 06
This week, our task was to classify images from the Cifar10 dataset.
To do so, we created a ResNet and a DenseNet network and compared them to our Simple CNN model from last week.
Also, we evaluated the models with different parameters to reach >85% accuracy.

## Dataset
We used the Tensorflow Cifar10 dataset. The dataset contains 50,000 training and 10,000 test images with shapes (32,32,3).
Pixel values range from 0 to 255.
Each image belongs to one of 10 labels.

*Reference:*
Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
URL: https://www.cs.toronto.edu/~kriz/cifar.html

## Models
We implemented a costum ResNet and a costum DenseNet model and reused our simple CNN model from last week.

## Training
For training we used Adam Optimizer and Categorical Crossentropy Loss.
We trained for 30 epochs with a learning rate of 0.001.

## Findings
**ResNet**

|   |   |   |   |
|---|---|---|---|
|Number of blocks   | 3  | 10 | 20 |
|Accuracy (after 10 epochs)   |  71% |  |  |
|Accuracy (after 30 epochs)   |  74% |  |  |
|Total params   | 217,034  |  |  |


**DenseNet**

|   |   |   |   |
|---|---|---|---|
|Growth Rate  | 2  | 4 |  |
|Accuracy (after 10 epochs)   | 79.66%  |  |  |
|Accuracy (after 30 epochs)   |  81.79% |  |  |
|Total params   | 4,726,890  |  |  |

**Simple CNN**

|   |   |
|---|---|
|Accuracy (after 10 epochs)   |   |
|Accuracy (after 30 epochs)   |   |
|Total params   |   |
