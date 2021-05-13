import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, dataloader
import sys, os
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise/cnn_cifar10')
from model.train_save_model import train_eval_model_gpu
from random_seed import fix_randomness
from basicblock import RecoverConv2d
from vgg_recover import recovered_net

def drive():
    
    fix_randomness(123)

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10('./data', train = True, download = True, transform = train_transform)
    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True, transform = test_transform)

    train_loader = DataLoader(train_set, batch_size = 50, shuffle = True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size = 50, shuffle = False, num_workers=2)

    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    device = torch.device(2)

    print('Run target model...')
    recover_model = recovered_net(conv_layers, 'W', True).to(device)

    train_eval_model_gpu(recover_model, 5, device, train_loader, test_loader, False, None)

if __name__ == '__main__':
    drive()
