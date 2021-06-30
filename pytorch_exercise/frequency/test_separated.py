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
from vgg_recover import recovered_net
from about_image import AddGaussianNoise

def drive():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    print('No inception, 5x5 Conv, MaxPool2d, no dropout, 0.25-weighted on boundary & ensemble network.')
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R']
    boundary_layers = [64, 128, 256, 512]

    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
    device = torch.device(2)

    #print('target(0.0), 224x224 STL10, random seed : 42, cam-layer : first MaxPool2d and RecoverConv2d')
    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
        #recover_model = AlexNet(True, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = separated_network(conv_layers, boundary_layers, device).to(device)


    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2238)),
        #AddGaussianNoise(0., 1., 0.3),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2238)),
        #AddGaussianNoise(0., 1., 0.3),
    ])

    train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size = 32, shuffle = True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers=2)

    train_save_model.train_eval_model_gpu(recover_model, 48, device, train_loader, test_loader, False, None)

def evaluate():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R', 512, 512, 'R']
    boundary_layers = [64, 128, 256, 512, 512]
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device(0)

    print('target model, no sum factor, no inception module, ImageNet subset (55 classes, train image : 71159, test_image : 2750)')
    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = separated_network(conv_layers, boundary_layers, device).to(device)
        recover_model.load_state_dict(torch.load('D:\\ImageNet\\separated_imagenet_noinception_subsetsum.pth', map_location="cuda:0"))


    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    test_set = torchvision.datasets.ImageFolder(root = 'D:\\imagenet_object_localization_patched2019\\ILSVRC\Data\\CLS-LOC\\valid_subset_sum', transform=test_transform)

    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers=2)

    ground_truth, eval_ret, boundary_ret, ensemble_ret = torch.empty(size = (2750, 1), dtype=torch.int), torch.empty(size = (2750, 1), dtype=torch.int), torch.empty(size = (2750, 1), dtype=torch.int), torch.empty(size = (2750, 1), dtype=torch.int)

    with torch.no_grad():

        valid_acc, boundary_acc, ensemble_acc = 0., 0., 0.
        
        for idx, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)

            recover_model.eval()

            output, boundary_output, ensemble_output = recover_model(data)

            _, pred = torch.max(output, dim = 1)
            _, boundray_pred = torch.max(boundary_output, dim = 1)
            _, ensemble_pred = torch.max(ensemble_output, dim = 1)
            
            ground_truth[idx] = target.item()
            eval_ret[idx] = pred.item()
            boundary_ret[idx] = boundray_pred.item()
            ensemble_ret[idx] = ensemble_pred.item()

            valid_acc += (pred == target)
            boundary_acc += (boundray_pred == target)
            ensemble_acc += (ensemble_pred == target)

            print(f'{idx+1}/{len(test_loader)} is completed\r', end = '')

    print('pred acc : {0:.4f}%, boundary acc : {1:.4f}%, ensemble acc : {2:.4f}%'.format(valid_acc/len(test_loader), boundary_acc/len(test_loader), ensemble_acc/len(test_loader)))
    print(classification_report(ground_truth.detach().cpu().numpy(), ensemble_ret.detach().cpu().numpy()))

def test():
    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R', 512, 512, 'R']
    boundary_layers = [64, 128, 256, 512, 512]
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device(0)

    print('target model, no sum factor, no inception module, ImageNet subset (55 classes, train image : 71159, test_image : 2750)')
    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = separated_network(conv_layers, boundary_layers, device).to(device)
        #recover_model.load_state_dict(torch.load('D:\\ImageNet\\separated_imagenet_noinception_subsetsum.pth', map_location="cuda:0"))

    print(summary(recover_model, (3, 224, 224)))

if __name__ == '__main__':
    drive()
    #evaluate()
    #test()