import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vgg_recover import recovered_net
from machine_learning.pytorch_exercise.cnn_cifar10.random_seed import fix_randomness
from machine_learning.pytorch_exercise.cnn_cifar10.model import train_save_model
from machine_learning.pytorch_exercise.cnn_cifar10.cam import grad_cam
from machine_learning.pytorch_exercise.cnn_cifar10.about_image import tensor_to_numpy
from machine_learning.pytorch_exercise.cnn_cifar10.about_image import inverse_normalize
import time


def drive():

    fix_randomness(123)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device('cuda')

    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
        recover_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\baseline_0427.pth', map_location="cuda:0"))
    else :
        print('Run target model...')
        recover_model = recovered_net(conv_layers, 'W', True).to(device)
        recover_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\recovered_upsample_0427.pth', map_location="cuda:0"))



    train_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10('./data', train = True, download = True,  transform=train_transform)
    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True,  transform=test_transform)

    train_loader = DataLoader(train_set, batch_size = 50, shuffle = True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size = 50, shuffle = False, num_workers=2)    


    image, label = iter(train_loader).next()
    sample, target = image[:8].to(device), label[:8].to(device)
    
    cam = grad_cam.grad_cam(recover_model)

    ret_cam, _ = cam.get_cam(sample, target)
    ret_cam = ret_cam.detach().cpu()
    sample = sample.detach().cpu()

    sample_denorm = inverse_normalize(sample, mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010), batch = True)

    ret_cam = tensor_to_numpy(ret_cam.unsqueeze(1), True)
    sample_denorm = tensor_to_numpy(sample_denorm, True)


    fig = plt.figure(figsize = (9, 9))
    for i in range(8):
        ax1 = fig.add_subplot(4, 4, 2*i+1)
        ax1.imshow(sample_denorm[i])
        ax2 = fig.add_subplot(4, 4, 2*i+2)
        ax2.imshow(ret_cam[i], cmap = 'jet')
    plt.savefig('./upsample_fix_cam')
    plt.show()

if __name__ == '__main__':
    drive()
    time.sleep(2)
    

