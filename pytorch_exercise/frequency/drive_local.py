import sys, os

from numpy.lib.arraysetops import isin
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise/cnn_cifar10')
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise')
sys.path.append('/home/sjlee/git_project/machine_learning')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\cnn_cifar10')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise')
# sys.path.append('C:\\anaconda3\envs\\torch\machine_learning')
from cam.grad_cam import grad_cam
from basicblock import RecoverConv2d
import torch
import time
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
from separated import separated_network
import torch.utils.model_zoo
from torchvision import models

def drive():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512,'R', 512, 512, 'R']
    boundary_layers = [64, 128, 256, 512, 512]
    #conv_layers = [63, 'R', 129, 'R', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #baseline_layers = [63, 63, 'M', 129, 129, 'M', 255, 255, 255, 'M', 513, 513, 513, 'M', 513, 513, 513, 'M']
    n_device = 1
    device = torch.device(n_device)

    print(time.strftime('%c', time.localtime(time.time())), ' device :', n_device)
    #print('Pretrained ImageNet, VGG16 based ensemble model')
    #print('target model, ensemble-fc-layer : 2048, 1.0-weight on backbone, 0.25-weight on boundary & ensemble, concat on feature-map, relu on concat')
    #print('VGG19 based model / ImageNet subset (55 classes, train image : 71159, test_image : 2750)')
    #print('saved as separated_ensemble_relu_vgg19_2048_1_5.pth')
    #print('baseline on subset-sum')

    pretrained = True
    subset = True

    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
        #recover_model = AlexNet(True, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = separated_network(conv_layers, boundary_layers, device)
        #recover_model = parallel_net(conv_layers, 'W', True, device).to(device)
        #recover_model = parallel_net(False, 'W', True, device).to(device)

    if pretrained :

        pretrained_param, dict_key_conv, dict_key_bn, dict_key_linear = download_params()

        recover_model = put_parameter(recover_model, pretrained_param, dict_key_conv, dict_key_bn, dict_key_linear)

        print('loading pretrained parameter is completed.')

    recover_model = recover_model.to(device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2238)),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2214, 0.2238)),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    if not subset :
        print('Run total ImageNet dataset')
        train_set = torchvision.datasets.ImageFolder(root = '/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train_subset_sum', transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root = '/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/val_subset_sum', transform=test_transform)

        train_loader = DataLoader(train_set, batch_size = 32, shuffle = True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers=2)

        print('Data load is completed...')

        train_save_model.train_eval_model_gpu(recover_model, 48, device, train_loader, test_loader, False, None)

    else :
        print('Run subset of ImageNet dataset')
        train_set = torchvision.datasets.ImageFolder(root = '/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/train_subset_sum', transform=train_transform)
        test_set = torchvision.datasets.ImageFolder(root = '/home/NAS_mount/sjlee/ILSVRC/Data/CLS-LOC/val_subset_sum', transform=test_transform)

        train_loader = DataLoader(train_set, batch_size = 32, shuffle = True, num_workers=2)
        test_loader = DataLoader(test_set, batch_size = 32, shuffle = False, num_workers=2)

        print('Data load is completed...')

        train_save_model.train_eval_model_gpu(recover_model, 48, device, train_loader, test_loader, False, None)



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
    device = torch.device(1)

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

def model_summary():
    fix_randomness(42)
    device = torch.device(0)
    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R', 512, 512, 'R']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    print('Run baseline model...')
    recover_model = recovered_net(baseline_layers, 'W', True).to(device)

    print(summary(recover_model, (3, 224, 224)))

def download_params():

    state_dict = dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'))

    dict_key_conv = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13']
    dict_key_bn = ['bn1', 'bn2', 'bn3', 'bn4', 'bn5', 'bn6', 'bn7', 'bn8', 'bn9', 'bn10', 'bn11', 'bn12', 'bn13']
    dict_key_linear = ['fc1', 'fc2', 'fc3']
    state_dict_keys = state_dict.keys()

    pretrained_param = {}
    pretrained_param['conv'] = {}
    pretrained_param['bn'] = {}
    pretrained_param['linear'] = {}

    for idx, k in enumerate(state_dict_keys):

        dict_idx = idx // 6
        weight_idx = idx % 6
        fc_dict_idx = (idx-78) // 2
        fc_weight_idx = idx % 2
        
        if idx < 78 :
                
            if weight_idx == 0 :
                pretrained_param['conv'][dict_key_conv[dict_idx]] = {}
                pretrained_param['conv'][dict_key_conv[dict_idx]]['weight'] = state_dict[k]
            elif weight_idx == 1 :
                pretrained_param['conv'][dict_key_conv[dict_idx]]['bias'] = state_dict[k]
            elif weight_idx == 2 :
                pretrained_param['bn'][dict_key_bn[dict_idx]] = {}
                pretrained_param['bn'][dict_key_bn[dict_idx]]['weight'] = state_dict[k]
            elif weight_idx == 3 :
                pretrained_param['bn'][dict_key_bn[dict_idx]]['bias'] = state_dict[k]
            elif weight_idx == 4 :
                pretrained_param['bn'][dict_key_bn[dict_idx]]['running_mean'] = state_dict[k]
            elif weight_idx == 5 :
                pretrained_param['bn'][dict_key_bn[dict_idx]]['running_var'] = state_dict[k]
            else :
                print('Error is occured!')
                return

        else :
            if fc_weight_idx == 0 :
                pretrained_param['linear'][dict_key_linear[fc_dict_idx]] = {}
                pretrained_param['linear'][dict_key_linear[fc_dict_idx]]['weight'] = state_dict[k]
            elif fc_weight_idx == 1 :
                pretrained_param['linear'][dict_key_linear[fc_dict_idx]]['bias'] = state_dict[k]
            else :
                print('Error is occured!')
                return
    
    return pretrained_param, dict_key_conv, dict_key_bn, dict_key_linear


def put_parameter(model, param_dict, dict_key_conv, dict_key_bn, dict_key_linear):

    conv_idx, bn_idx, fc_idx = 0, 0, 0

    for m in model.features.modules():

        if isinstance(m, nn.Conv2d) :
            with torch.no_grad():
                m.weight = nn.Parameter(param_dict['conv'][dict_key_conv[conv_idx]]['weight'])
                m.bias = nn.Parameter(param_dict['conv'][dict_key_conv[conv_idx]]['bias'])
                #print(dict_key_conv[conv_idx], 'is setted.')
                conv_idx += 1
        
        elif isinstance(m, nn.BatchNorm2d) :
            with torch.no_grad():
                m.weight = nn.Parameter(param_dict['bn'][dict_key_bn[bn_idx]]['weight'])
                m.bias = nn.Parameter(param_dict['bn'][dict_key_bn[bn_idx]]['bias'])
                m.running_mean = param_dict['bn'][dict_key_bn[bn_idx]]['running_mean']
                m.running_var = param_dict['bn'][dict_key_bn[bn_idx]]['running_var']
                #print(dict_key_bn[bn_idx], 'is setted.')
                bn_idx += 1

    for m in model.classifier.modules():
        print(m.weight.shape)
        print(m.bias.shape)
        print(param_dict['linear'][dict_key_linear[fc_idx]]['weight'].shape)
        print(param_dict['linear'][dict_key_linear[fc_idx]]['bias'].shape)
        if isinstance(m, nn.Linear) :
            with torch.no_grad():
                m.weight = nn.Parameter(param_dict['linear'][dict_key_linear[fc_idx]]['weight'])
                m.bias = nn.Parameter(param_dict['linear'][dict_key_linear[fc_idx]]['bias'])
                #print(dict_key_linear[fc_idx], 'is setted.')
                fc_idx += 1

    return model


if __name__ == '__main__':
    drive()
    #test()
    #model_summary()
