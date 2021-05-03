import torch
import numpy as np

def tensor_to_numpy(tensor, batch = False):
    image = tensor.numpy()
    if batch :
        image_tr = np.transpose(image, (0, 2, 3, 1))
    else :
        image_tr = np.transpose(image, (1, 2, 0))
    return image_tr

def inverse_normalize(image, mean, std, batch = False):
    ret_image = torch.empty(image.shape)
    if batch:
        for i in range(3):
            ret_image[:, i, :, :] = (image[:, i, :, :]*std[i]) + mean[i]
    else:
        for i in range(3):
            ret_image[i, :, :] = (image[i, :, :]*std[i]) + mean[i]
    return ret_image