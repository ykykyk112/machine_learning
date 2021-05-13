import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, names, filename, save_path = 'pytorch_exercise/cnn_cifar10/model_saved/', title='Confusion matrix', cmap=plt.cm.OrRd):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 'xx-large')
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path + filename + '.jpg')