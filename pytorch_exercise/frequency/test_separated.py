import sys, os
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise/cnn_cifar10')
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise')
sys.path.append('/home/sjlee/git_project/machine_learning')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\cnn_cifar10')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning')
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
from separated import separated_network

def drive():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R']
    boundary_layers = [64, 128, 256, 512]

    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device(2)

    #print('target(0.0), 224x224 STL10, random seed : 42, cam-layer : first MaxPool2d and RecoverConv2d')
    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
        #recover_model = AlexNet(True, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = separated_network(conv_layers, boundary_layers).to(device)


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root = './data', dowmload = True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root = './data', download = True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size = 32, shuffle = True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers=2)

    train_save_model.train_eval_model_gpu(recover_model, 36, device, train_loader, test_loader, False, None)