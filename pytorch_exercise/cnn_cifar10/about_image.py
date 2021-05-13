import torch
import numpy as np
import matplotlib.colors as colors

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

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))