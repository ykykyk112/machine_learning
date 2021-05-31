import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def make_dataset():
    proxy = nib.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\Task06_Lung\imagesTr\lung_001.nii.gz')
    label = nib.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\Task06_Lung\labelsTr\lung_001.nii.gz')
    
    image = proxy.dataobj[:, :, 235]
    target = label.dataobj[:, :, 235]

    if 1.0 in target:
        print('True')
    else:
        print('False')

    window_center, window_width = -600, 1600
    image3 = np.clip(image, window_center - (window_width / 2), window_center + (window_width / 2))
    print(image3.max(), image3.min())
    image3_re = (image3 - image3.min()) / (image3.max() - image3.min())
    print(image3_re.max(), image3_re.min())
    plt.imshow(image3_re, cmap = 'gray')
    plt.imshow(target, cmap = 'jet', alpha = 0.3)
    plt.show()

if __name__ == '__main__' :
    make_dataset()