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
from about_image import inverse_normalize


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
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True,  transform=test_transform)
    test_loader = DataLoader(test_set, batch_size = 2, shuffle = False, num_workers=2)   

    recover_model.eval()
    
    valid_loss, valid_acc = 0., 0.

    prediction = torch.empty()

    for idx, (valid_data, valid_target) in enumerate(test_loader) :
                
        valid_data, valid_target = valid_data.to(device), valid_target.to(device)


        valid_output = recover_model(valid_data, valid_target, idx, True)


        v_loss = recover_model.loss(valid_output, valid_target)
        _, v_pred = torch.max(valid_output, dim = 1)

        valid_loss += v_loss.item()
        valid_acc += torch.sum(v_pred == valid_target.data)
        print(idx, '/', len(test_loader))
    
    #valid_acc = valid_acc*(100/valid_data.size()[0])

    print('loss', valid_loss, 'accuracy', valid_acc)

def plot_cam():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    device = torch.device(0)

    conv_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'R', 512, 512, 'R']

    print('Run target model...')
    recover_model = parallel_net(conv_layers, 'W', True, device).to(device)
    recover_model.load_state_dict(torch.load('pytorch_exercise\\frequency\data\\target_parameter.pth'))
    recover_model.latest_valid_cam = valid_cam = torch.tensor(np.load('pytorch_exercise\\frequency\data\\cam_ret.npy')).to(device)
    print('complete load all parameters')
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True,  transform=test_transform)
    test_loader = DataLoader(test_set, batch_size = 10000, shuffle = False, num_workers=0)
    
    image, label = iter(test_loader).next()
    image_denorm = inverse_normalize(image, mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010), batch = True)

    valid_cam = torch.tensor(np.load('pytorch_exercise\\frequency\data\\cam_ret.npy'))

    print(valid_cam.shape)

    upsample = nn.Upsample(size = 224, mode = 'bilinear', align_corners = False)
    
    cam_upsample = upsample(valid_cam)

    image_np = np.transpose(image_denorm.numpy(), (0, 2, 3, 1))
    cam_np = np.transpose(cam_upsample.numpy(), (0, 2, 3, 1))

    fig = plt.figure(figsize = (12, 12))

    index = 45

    for i in range(3):
        for j in range(3):
            ax = fig.add_subplot(3, 3, 3*i+j+1)
            ax.imshow(image_np[3*i+j+1+index])
            ax.imshow(cam_np[3*i+j+1+index], cmap = 'jet', alpha = 0.3)
            ax.axis('off')
    plt.show()


if __name__ == '__main__' :
    #make_dataset()
    put_parameters()
    #plot_cam()