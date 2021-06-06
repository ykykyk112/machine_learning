import enum
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import time

from torch.functional import split

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
from frequency.basicblock import RecoverConv2d
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

    prediction = torch.empty((10000, ))

    for idx, (valid_data, valid_target) in enumerate(test_loader) :
                
        valid_data, valid_target = valid_data.to(device), valid_target.to(device)


        valid_output = recover_model(valid_data, valid_target, idx, True)

        v_loss = recover_model.loss(valid_output, valid_target)
        _, v_pred = torch.max(valid_output, dim = 1)

        valid_loss += v_loss.item()
        valid_acc += torch.sum(v_pred == valid_target.data)
        print(idx, '/', len(test_loader))
        prediction[idx*2:(idx+1)*2] = v_pred
    
    #valid_acc = valid_acc*(100/valid_data.size()[0])

    print('loss', valid_loss, 'accuracy', valid_acc)

    np.save('./model_pred.npy', prediction)

def plot_cam():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    device = torch.device(0)

    conv_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'R', 512, 512, 'R']
    conv_layers_2 = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R', 512, 512, 'R']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    print('Run baseline model...')
    half_model = recovered_net(conv_layers_2, 'W', True).to(device)
    #half_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\adaptive_123.pth', map_location="cuda:0"))

    baseline_model = recovered_net(baseline_layers, 'W', True).to(device)
    #baseline_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\baseline_123.pth', map_location="cuda:0"))


    one_model = recovered_net(conv_layers_2, 'W', True).to(device)
    #one_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\adaptive_1_123.pth', map_location="cuda:0"))

    print('Run target model...')
    recover_model = parallel_net(conv_layers_2, 'W', True, device).to(device)
    #recover_model.load_state_dict(torch.load('pytorch_exercise\\frequency\data\\target_parameter_former.pth'))
    #recover_model.latest_valid_cam = valid_cam = torch.tensor(np.load('pytorch_exercise\\frequency\data\\cam_ret_former.npy')).to(device)
    print('complete load all parameters')
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True,  transform=test_transform)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=0)
    

    valid_cam = torch.tensor(np.load('pytorch_exercise\\frequency\data\\cam_ret_lowall.npy'))

    upsample = nn.Upsample(size = 224, mode = 'bilinear', align_corners = False)
    
    cam_upsample = upsample(valid_cam)
    cam_np = np.transpose(cam_upsample.numpy(), (0, 2, 3, 1))
    cam_pred = np.load('./model_pred.npy')

    baseline_cam = grad_cam(baseline_model, True)
    recover_cam = grad_cam(recover_model, False)
    half_cam = grad_cam(half_model, False)
    one_cam = grad_cam(one_model, False)

    class_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for idx, (test_data, test_target) in enumerate(test_loader):

        test_data, test_target = test_data.to(device), test_target.to(device)
        
        recover_model.eval()
        baseline_model.eval()
        one_model.eval()
        half_model.eval()

        image_denorm = inverse_normalize(test_data, mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010), batch = True)
        image_np = np.transpose(image_denorm.numpy(), (0, 2, 3, 1)).reshape(224, 224, 3)

        baseline_ret, baseline_pred = baseline_cam.get_cam(test_data, test_target)
        baseline_ret, baseline_pred = upsample(baseline_ret.detach().cpu().unsqueeze(0)), baseline_pred.detach().cpu()
        return
        one_ret, one_pred = one_cam.get_cam(test_data, test_target)
        one_ret, one_pred = upsample(one_ret.detach().cpu().unsqueeze(0)), one_pred.detach().cpu()

        #recover_ret, recover_pred = recover_cam.get_cam_for_recover(test_data, test_target, idx)
        #recover_ret, recover_pred = upsample(recover_ret.detach().cpu().unsqueeze(0)), recover_pred.detach().cpu()

        half_ret, half_pred = half_cam.get_cam(test_data, test_target)
        half_ret, half_pred = upsample(half_ret.detach().cpu().unsqueeze(0)), half_pred.detach().cpu()

        baseline_np, recover_np, one_np, half_np = baseline_ret.numpy().reshape(224, 224, 1), one_ret.numpy().reshape(224, 224, 1), one_ret.numpy().reshape(224, 224, 1), half_ret.numpy().reshape(224, 224, 1)
        
        #if not( int(baseline_pred)==int(half_pred) and int(half_pred)==int(one_pred) and int(one_pred) == int(test_target) and int(test_target) == int(baseline_pred)):
        #    print('Error case')
        #    continue

        # if int(test_target) != 3:
        #     continue

        fig = plt.figure(figsize=(16, 4))

        ax1 = fig.add_subplot(1, 5, 1)
        ax1.imshow(image_np)
        ax1.set_title(class_map[int(test_target.detach().cpu())])
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 5, 2)
        ax2.imshow(image_np)
        ax2.imshow(baseline_np, cmap = 'jet', alpha = 0.3)
        ax2.set_title('Baseline')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(1, 5, 3)
        ax3.imshow(image_np)
        ax3.imshow(half_np, cmap = 'jet', alpha = 0.3)
        ax3.set_title('Recover(1.0)')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(1, 5, 4)
        ax4.imshow(image_np)
        ax4.imshow(one_np, cmap = 'jet', alpha = 0.3)
        ax4.set_title('Recover(0.5)')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(1, 5, 5)
        ax5.imshow(image_np)
        ax5.imshow(cam_np[idx], cmap = 'jet', alpha = 0.3)
        ax5.set_title('Heatmap mask')
        ax5.axis('off')

        plt.show()
        #plt.savefig('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\\frequency_cam\\both_correct2\\{0}\\{1}.png'.format(class_map[int(baseline_pred)], idx))
        #plt.close()
        print(idx, '/', len(test_loader))
        del test_data, test_target

def plot_maxpool_cam():
    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    device = torch.device(0)

    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R', 512, 512, 'R']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    print('Run baseline model...')
    baseline_model = recovered_net(baseline_layers, 'W', True).to(device)
    baseline_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\target_parameter_stl.pth', map_location="cuda:0"))

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.STL10('./data', split = 'test', download = True,  transform=test_transform)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=0)

    baseline_cam = grad_cam(baseline_model, False)
    maxpool_cam = grad_cam(baseline_model, True)

    class_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    valid_cam = torch.tensor(np.load('pytorch_exercise\\frequency\data\\cam_ret_stl.npy'))
    upsample = nn.Upsample(size = 224, mode = 'bilinear', align_corners = False)
    
    recover_ret = upsample(valid_cam)

    for idx, (test_data, test_target) in enumerate(test_loader):

        test_data, test_target = test_data.to(device), test_target.to(device)
        
        baseline_model.eval()

        image_denorm = inverse_normalize(test_data, mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010), batch = True)
        image_np = np.transpose(image_denorm.numpy(), (0, 2, 3, 1)).reshape(224, 224, 3)

        baseline_ret, baseline_pred = baseline_cam.get_cam_for_recover(test_data, test_target, idx)
        baseline_ret, baseline_pred = upsample(baseline_ret.detach().cpu().unsqueeze(0)), baseline_pred.detach().cpu()

        maxpool_ret, maxpool_pred = maxpool_cam.get_cam(test_data, test_target)
        maxpool_ret, maxpool_pred = upsample(maxpool_ret.detach().cpu().unsqueeze(0)), maxpool_pred.detach().cpu()

        baseline_np = baseline_ret.numpy().reshape(224, 224, 1)
        recover_np = recover_ret[idx].numpy().reshape(224, 224, 1)

        fig = plt.figure(figsize=(12, 4))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(image_np)
        ax1.set_title(class_map[int(test_target.detach().cpu())])
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(image_np)
        ax2.imshow(baseline_np, cmap = 'jet', alpha = 0.4)
        ax2.set_title('Baseline')
        ax2.axis('off')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(image_np)
        ax3.imshow(recover_np, cmap = 'jet', alpha = 0.4)
        ax3.set_title('Recover')
        ax3.axis('off')

        plt.show()
        print(idx/len(test_loader))

def get_fam():

    seed_number = 42
    print('seed number :', seed_number)
    fix_randomness(seed_number)
    
    device = torch.device(0)

    conv_layers = [64, 'R', 128, 'R', 256, 256, 'R', 512, 512, 'R', 512, 512, 'R']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    print('Run baseline model...')
    baseline_model = recovered_net(conv_layers, 'W', True).to(device)
    baseline_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\target_parameter_cifar_224.pth', map_location="cuda:0"))

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True,  transform=test_transform)
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers=0)

    #class_map = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    class_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    baseline_model.register_hook()

    upsample = nn.Upsample(size = 224, mode = 'bilinear', align_corners=False)

    for idx, (test_data, test_target) in enumerate(test_loader):

        test_data, test_target = test_data.to(device), test_target.to(device)
        
        baseline_model.eval()

        image_denorm = inverse_normalize(test_data, mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010), batch = True)
        image_np = np.transpose(image_denorm.numpy(), (0, 2, 3, 1)).reshape(224, 224, 3)

        output = baseline_model(test_data)
        _, pred = torch.max(output, dim = 1)

        f_map_boundary = torch.zeros((1, 1, 224, 224)).to(device)
        f_map = torch.zeros((1, 1, 224, 224)).to(device)

        for idx, f in enumerate(baseline_model.feature_maps):
            # origin feature map sum
            if idx%2 == 0:
                f_sum = torch.sum(f, dim = (1), keepdim = True)
                f_upsample = upsample(f_sum)
                f_map += (f_upsample/f.size(1))
            # boundary feature map sum
            else:
                f_sum = torch.sum(f, dim = (1), keepdim = True)
                f_upsample = upsample(f_sum)
                f_map_boundary += (f_upsample/f.size(1))
            f_np = f_upsample.detach().cpu().numpy().reshape(224, 224, 1)
            #plt.imshow(f_np, cmap = 'jet')
            #plt.title(class_map[int(pred)])
            #plt.show()

        f_map_boundary_np = f_map_boundary.detach().cpu().numpy().reshape(224, 224, 1)
        f_min, f_max = np.min(f_map_boundary_np), np.max(f_map_boundary_np)
        f_map_boundary_np = (f_map_boundary_np - f_min) / (f_max - f_min)
        f_map_np = f_map.detach().cpu().numpy().reshape(224, 224, 1)
        f_total = f_map_boundary_np + f_map_np

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        ax1.imshow(image_np)
        ax1.set_title(class_map[int(pred)])
        ax1.axis('off')

        ax2.imshow(f_map_np, cmap = 'jet')
        ax2.set_title('feature activation mapping')
        ax2.axis('off')

        ax3.imshow(f_map_boundary_np, cmap = 'jet')
        ax3.set_title('boudnary activation mapping')
        ax3.axis('off')
        
        ax4.imshow(f_total, cmap = 'jet')
        ax4.set_title('total activation mapping')
        ax4.axis('off')
        
        plt.show()
        



if __name__ == '__main__' :
    #make_dataset()
    #put_parameters()
    #plot_cam()
    get_fam()