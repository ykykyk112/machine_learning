import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\cnn_cifar10')
sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise')
sys.path.append('C:\\anaconda3\envs\\torch\machine_learning')
from cam.grad_cam import grad_cam
from basicblock import RecoverConv2d
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary
from random_seed import fix_randomness
from model import train_save_model
from parallel import parallel_net
from frequency.vgg_recover import recovered_net
from alexnet import AlexNet


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

def put_parameters():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    conv_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'R', 512, 512, 'R']
    #conv_layers = [63, 'R', 129, 'R', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #baseline_layers = [63, 63, 'M', 129, 129, 'M', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    device = torch.device(0)

    print('no dropout, default value : 1.0, random crop, train - target, validation - pred, 224x224')
    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = parallel_net(conv_layers, 'W', True, device).to(device)
        recover_model.load_state_dict(torch.load('pytorch_exercise\\frequency\data\\target_parameter.pth'))
        valid_cam = torch.tensor(np.load('pytorch_exercise\\frequency\data\\cam_ret.npy')).to(device)
        recover_model.latest_valid_cam = valid_cam
        print('complete load all parameters')

    print(recover_model.latest_valid_cam)

if __name__ == '__main__' :
    #make_dataset()
    put_parameters()