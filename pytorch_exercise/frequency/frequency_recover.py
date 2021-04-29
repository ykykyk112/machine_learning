import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vgg_recover import recovered_net
from torchsummary import summary
from machine_learning.pytorch_exercise.cnn_cifar10.random_seed import fix_randomness
from machine_learning.pytorch_exercise.cnn_cifar10.model import train_save_model

def drive():

    fix_randomness(1)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device(0)

    recover_model = recovered_net(conv_layers, 'W', True).to(device)

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

    train_set = torchvision.datasets.CIFAR10('data', train = True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10('data', train = False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size = 50, shuffle = True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size = 50, shuffle = False, num_workers=2)    

    train_save_model.train_eval_model_gpu(recover_model, 25, device, train_loader, test_loader, False)

if __name__ == '__main__':
    drive()