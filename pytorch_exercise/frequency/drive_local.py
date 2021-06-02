import sys, os
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise/cnn_cifar10')
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise')
sys.path.append('/home/sjlee/git_project/machine_learning')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\cnn_cifar10')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning')
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

def drive():


    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R', 512, 512, 'R']
    #conv_layers = [63, 'R', 129, 'R', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #baseline_layers = [63, 63, 'M', 129, 129, 'M', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    device = torch.device(1)

    print('baseline, 224x224, STL10, random seed : 42')
    if not False:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
        #recover_model = AlexNet(True, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = parallel_net(conv_layers, 'W', True, device).to(device)
        #recover_model = parallel_net(False, 'W', True, device).to(device)


    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.STL10(root = './data', split='train', download = True,  transform=train_transform)
    test_set = torchvision.datasets.STL10(root = './data', split = 'test', download = True,  transform=test_transform)

    train_loader = DataLoader(train_set, batch_size = 50, shuffle = True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size = 50, shuffle = False, num_workers=2)


    train_save_model.train_eval_model_gpu(recover_model, 75, device, train_loader, test_loader, False, None)



def test():

    fix_randomness(42)

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True, transform = test_transform)

    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers=2)

    sample, label = iter(test_loader).next()

    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #conv_layers = [63, 'R', 129, 'R', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #baseline_layers = [63, 63, 'M', 129, 129, 'M', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    device = torch.device(0)

    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = recovered_net(conv_layers, 'W', True)
        recover_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\adaptive_1_123.pth', map_location="cuda:0"))

    cam = grad_cam(recover_model)

    batch_cam = cam.get_batch_label_cam(sample, label)
    label_cam, _ = cam.get_label_cam(sample, label)

    batch_cam = batch_cam.unsqueeze(3).detach().numpy()
    label_cam = label_cam.unsqueeze(3).detach().numpy()

    print(batch_cam.shape, label_cam.shape)

    sample_np = np.transpose(sample.detach().cpu().numpy(), (0, 2, 3, 1))
    
    fig = plt.figure(figsize = (15, 15))
    for i in range(3):
        ax1 = fig.add_subplot(3, 3, i*3+1)
        ax1.imshow(sample_np[i])
        ax2 = fig.add_subplot(3, 3, i*3+2)
        ax2.imshow(batch_cam[i], cmap = 'jet')
        ax3 = fig.add_subplot(3, 3, i*3+3)
        ax3.imshow(label_cam[i], cmap = 'jet')

    plt.show()


if __name__ == '__main__':
    drive()
    #test()
