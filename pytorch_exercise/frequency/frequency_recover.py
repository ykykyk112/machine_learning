import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from vgg_recover import recovered_net
from torchsummary import summary
import argparse
import sys, os
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise/cnn_cifar10')
from random_seed import fix_randomness
from model import train_save_model

def drive():


    fix_randomness(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='W')
    parser.add_argument('--upsample', default=True, type = lambda s : s == 'True')
    parser.add_argument('--device', default = 0, type=int)
    parser.add_argument('--baseline', default=False, type = lambda s : s == 'False')
    args = parser.parse_args()
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device(args.device)

    if not args.baseline:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, args.mode, args.upsample).to(device)
    else :
        print('Run target model...')
        recover_model = recovered_net(conv_layers, args.mode, args.upsample).to(device)

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

    train_save_model.train_eval_model_gpu(recover_model, 25, device, train_loader, test_loader, False)

if __name__ == '__main__':
    drive()
