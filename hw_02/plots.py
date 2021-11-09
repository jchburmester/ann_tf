# -*- coding: utf-8 -*-
"""
Python script for plotting
created on 03.11.2021 by jchburmester

"""

import numpy as np
import matplotlib.pyplot as plt
from functions import sigmoid, sigmoidprime
from training import epoch_loss
from training import epoch_accuracy

# plotting sigmoids
x = np.arange(-8.0, 8.0, 0.01)
y = sigmoid(x)
z = sigmoidprime(x)

fig1, ax1 = plt.subplots()
ax1.plot(x, y)
ax1.set(xlabel='x', ylabel='sigmoid(x)',
       title='sigmoid')
ax1.grid()
fig1.savefig("sigmoid.png")
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(x, z)
ax2.set(xlabel='x', ylabel='sigmoidprime(x)',
       title='sigmoidprime')
ax2.grid()
fig2.savefig("sigmoid.png")
plt.show()

# plotting accuracy and loss after training MLP
x = np.arange(1000)
plt.plot(x, epoch_loss)
plt.plot(x, epoch_accuracy)
plt.show()