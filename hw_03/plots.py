# -*- coding: utf-8 -*-
"""
Python script for visualisation
created on 03.11.2021

"""

import matplotlib.pyplot as plt
import numpy as np
from training import accuracies, losses, train_losses

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0,len(accuracies)), accuracies, label='Accuracy')
plt.plot(np.arange(0,len(losses)), losses, label='Loss')
plt.plot(np.arange(0,len(train_losses)), train_losses, label='Train loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.show()