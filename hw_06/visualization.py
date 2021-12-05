import matplotlib.pyplot as plt

def visualize(results):
    """
    Visualizes training, validation and test accuracy and loss

    Parameters:
    ----------
    results : list
        list with training losses, val losses, val accuracies, test loss, test accuracy
    """
plt.figure()
plt.plot(results[0], color='mediumturquoise', label='train loss')
plt.plot(results[1], color='navy', label='validation loss')
plt.plot(results[2], color='peru', label='validation accuracy')
plt.axhline(results[3], color='tomato', label='test loss')
plt.axhline(results[4], color='olive', label='test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
