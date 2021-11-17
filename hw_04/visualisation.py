import matplotlib as plt
import numpy as np
from main import accuracies, losses, train_losses

''' Visualize accuracy and loss for training and test data '''

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0,len(accuracies)), accuracies, label='Accuracy')
plt.plot(np.arange(0,len(losses)), losses, label='Loss')
plt.plot(np.arange(0,len(train_losses)), train_losses, label='Train loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.show()