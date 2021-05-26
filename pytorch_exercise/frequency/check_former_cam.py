import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, dataloader
import sys, os
sys.path.append('/home/sjlee/git_project/machine_learning/pytorch_exercise/cnn_cifar10')
from machine_learning.pytorch_exercise.cnn_cifar10.random_seed import fix_randomness
from machine_learning.pytorch_exercise.cnn_cifar10.model import custom_dataset
from machine_learning.pytorch_exercise.cnn_cifar10.cam.grad_cam import grad_cam
from machine_learning.pytorch_exercise.cnn_cifar10.about_image import tensor_to_numpy
from machine_learning.pytorch_exercise.cnn_cifar10.about_image import inverse_normalize
from machine_learning.pytorch_exercise.frequency.vgg_recover import recovered_net
from machine_learning.pytorch_exercise.cnn_cifar10.model import train_save_model
from machine_learning.pytorch_exercise.cnn_cifar10.about_image import MidpointNormalize



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

    train_save_model.train_eval_model_gpu(recover_model, 5, device, train_loader, test_loader, False, None)

def extract_figure():

    fix_randomness(123)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device('cuda')

    print('Run target model...')
    recover_model = recovered_net(conv_layers, 'W', True).to(device)
    recover_model.load_state_dict(torch.load('./recover_cam_5.pth', map_location="cuda:0"))

    path = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\5th_epoch\\'

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform = transform)
    test_loader = DataLoader(test_set, batch_size=50, shuffle=False, num_workers=2)
    
    sample, label = iter(test_loader).next()

    recover_model = recover_model.to('cpu')
    recover_cam = grad_cam.grad_cam(recover_model)

    # # Register hook & get former layer's activation map
    ret_cam, _ = recover_cam.get_label_cam(sample, label)
    ret_downsample_cam, _ = recover_cam.get_downsample_label_cam(sample, label)
    ret_cam = ret_cam.detach()
    ret_downsample_cam = ret_downsample_cam.detach()

    np.save(path + 'original_cam.npy', ret_cam)
    np.save(path + 'downsample_cam.npy', ret_downsample_cam)

    with torch.no_grad():
        
        recover_model.eval()
        recover_model.register_hook()
        output = recover_model(sample)
        _, pred = torch.max(output, dim = 1)
        correction = pred==label
        print(correction)

        first_map = recover_model.feature_maps[2].detach().numpy()
        second_map = recover_model.feature_maps[5].detach().numpy()


    first_map_np = np.transpose(first_map, (0, 2, 3, 1))
    second_map_np = np.transpose(second_map, (0, 2, 3, 1))
    
    np.save(path + 'first_map.npy', first_map)
    np.save(path + 'second_map.npy', second_map)
    np.save(path + 'correction.npy', correction)

    sample_denorm = inverse_normalize(sample.detach(), (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), batch = True).numpy()
    sample_tr = np.transpose(sample_denorm, (0, 2, 3, 1))

    np.save(path + 'sample_image.npy', sample_tr)

def plot_activation_cam():

    path1 = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\1st_epoch\\'
    path2 = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\2nd_epoch\\'
    path3 = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\3rd_epoch\\'
    path4 = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\4th_epoch\\'
    path5 = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\5th_epoch\\'

    
    sample_image = np.load(path1+'sample_image.npy')

    path = [path1, path2, path3, path4, path5]
    correction, first_map, second_map, original_cam, downsample_cam = [], [], [], [], []

    for p in path:
        correction.append(np.load(p+'correction.npy'))
        
        first_map.append(np.transpose(np.load(p+'first_map.npy'), (0, 2, 3, 1)))
        second_map.append(np.transpose(np.load(p+'second_map.npy'), (0, 2, 3, 1)))
        
        original_cam.append(np.load(p+'original_cam.npy'))
        downsample_cam.append(np.load(p+'downsample_cam.npy'))

    path_idx = 0

    save_path = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\second_dot_cam\\'

    sc_1 = torch.FloatTensor(downsample_cam[0].reshape(50, 1, 14, 14))
    sc_2 = torch.FloatTensor(downsample_cam[1].reshape(50, 1, 14, 14))
    sc_3 = torch.FloatTensor(downsample_cam[2].reshape(50, 1, 14, 14))
    sc_4 = torch.FloatTensor(downsample_cam[3].reshape(50, 1, 14, 14))
    sc_5 = torch.FloatTensor(downsample_cam[4].reshape(50, 1, 14, 14))

    #upsample_112 = nn.Upsample(size = 112, mode = 'bilinear', align_corners = False)
    upsample_56 = nn.Upsample(size = 56, mode = 'bilinear', align_corners = False)

    # sc_112_1 = upsample_112(sc_1).numpy().reshape(50, 112, 112, 1)
    # sc_112_2 = upsample_112(sc_2).numpy().reshape(50, 112, 112, 1)
    # sc_112_3 = upsample_112(sc_3).numpy().reshape(50, 112, 112, 1)
    # sc_112_4 = upsample_112(sc_4).numpy().reshape(50, 112, 112, 1)
    # sc_112_5 = upsample_112(sc_5).numpy().reshape(50, 112, 112, 1)

    sc_56_1 = upsample_56(sc_1).numpy().reshape(50, 56, 56, 1)
    sc_56_2 = upsample_56(sc_2).numpy().reshape(50, 56, 56, 1)
    sc_56_3 = upsample_56(sc_3).numpy().reshape(50, 56, 56, 1)
    sc_56_4 = upsample_56(sc_4).numpy().reshape(50, 56, 56, 1)
    sc_56_5 = upsample_56(sc_5).numpy().reshape(50, 56, 56, 1)

    am_1 = second_map[0]
    am_2 = second_map[1]
    am_3 = second_map[2]
    am_4 = second_map[3]
    am_5 = second_map[4]

    for i in range(50):
        c_min1, c_max1 = sc_56_1[i].min(), sc_56_1[i].max()
        sc_56_1[i] = (sc_56_1[i] - c_min1) / (c_max1 - c_min1)

        c_min2, c_max2 = sc_56_2[i].min(), sc_56_2[i].max()
        sc_56_2[i] = (sc_56_2[i] - c_min2) / (c_max2 - c_min2)

        c_min3, c_max3 = sc_56_3[i].min(), sc_56_3[i].max()
        sc_56_3[i] = (sc_56_3[i] - c_min3) / (c_max3 - c_min3)

        c_min4, c_max4 = sc_56_4[i].min(), sc_56_4[i].max()
        sc_56_4[i] = (sc_56_4[i] - c_min4) / (c_max4 - c_min4)

        c_min5, c_max5 = sc_56_5[i].min(), sc_56_5[i].max()
        sc_56_5[i] = (sc_56_5[i] - c_min5) / (c_max5 - c_min5)


    for i in range(50):
        
        try :
            os.makedirs(save_path+str(i+1))
        except OSError:
            print('os makedirs error.')

        for j in range(128):

            fig = plt.figure(figsize=(15, 8))

            ax1 = fig.add_subplot(2, 5, 1)
            a1 = am_1[i, :, :, j].reshape(56, 56, 1)
            b1 = sc_56_1[i, :, :]
            c1 = a1*b1
            ax1.imshow(c1, cmap = 'Reds')
            ax1.set_title('1st epoch')
            ax1.axis('off')

            ax2 = fig.add_subplot(2, 5, 2)
            a2 = am_2[i, :, :, j].reshape(56, 56, 1)
            b2 = sc_56_2[i, :, :]
            c2 = a2*b2
            ax2.imshow(c2, cmap = 'Reds')
            ax2.set_title('2nd epoch')
            ax2.axis('off')

            ax3 = fig.add_subplot(2, 5, 3)
            a3 = am_3[i, :, :, j].reshape(56, 56, 1)
            b3 = sc_56_3[i, :, :]
            c3 = a3*b3
            ax3.imshow(c3, cmap = 'Reds')
            ax3.set_title('3rd epoch')
            ax3.axis('off')

            ax4 = fig.add_subplot(2, 5, 4)
            a4 = am_4[i, :, :, j].reshape(56, 56, 1)
            b4 = sc_56_4[i, :, :]
            c4 = a4*b4
            ax4.imshow(c4, cmap = 'Reds')
            ax4.set_title('4th epoch')
            ax4.axis('off')

            ax5 = fig.add_subplot(2, 5, 5)
            a5 = am_5[i, :, :, j].reshape(56, 56, 1)
            b5 = sc_56_5[i, :, :]
            c5 = a5*b5
            ax5.imshow(c5, cmap = 'Reds')
            ax5.set_title('5th epoch')
            ax5.axis('off')

            ax6 = fig.add_subplot(2, 5, 6)
            ax6.imshow(am_1[i, :, :, j].reshape(56, 56, 1), cmap = 'Reds')
            ax6.set_title('1st epoch')
            ax6.axis('off')

            ax7 = fig.add_subplot(2, 5, 7)
            ax7.imshow(am_2[i, :, :, j].reshape(56, 56, 1), cmap = 'Reds')
            ax7.set_title('2nd epoch')
            ax7.axis('off')

            ax8 = fig.add_subplot(2, 5, 8)
            ax8.imshow(am_3[i, :, :, j].reshape(56, 56, 1), cmap = 'Reds')
            ax8.set_title('3rd epoch')
            ax8.axis('off')

            ax9 = fig.add_subplot(2, 5, 9)
            ax9.imshow(am_4[i, :, :, j].reshape(56, 56, 1), cmap = 'Reds')
            ax9.set_title('4th epoch')
            ax9.axis('off')

            ax10 = fig.add_subplot(2, 5, 10)
            ax10.imshow(am_5[i, :, :, j].reshape(56, 56, 1), cmap = 'Reds')
            ax10.set_title('5th epoch')
            ax10.axis('off')

            plt.savefig(save_path+str(i+1)+'\\{}'.format(j+1))
            plt.close()


def check_derivative():
    
    path = 'C:\\anaconda3\\envs\\torch\\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\5th_epoch\\'

    image = np.load(path + 'sample_image.npy')
    feature_map = np.transpose(np.load(path + 'first_map.npy'), (0, 2, 3, 1))

    feature_map_left = np.zeros((50, 112, 112, 64))
    feature_map_up = np.zeros((50, 112, 112, 64))

    feature_map_left[:, :111, :, :] = feature_map[:, 1: ,:, :]
    feature_map_up[:, :, :111, :] = feature_map[:, :, 1:, :]

    idx = 12
    sample = 15

    feature_map_x_diff = np.abs(feature_map[sample, :, :, idx].reshape(112, 112, 1)-feature_map_left[sample, :, :, idx].reshape(112, 112, 1))
    feature_map_y_diff = np.abs(feature_map[sample, :, :, idx].reshape(112, 112, 1)-feature_map_up[sample, :, :, idx].reshape(112, 112, 1))

    diff_sqrt = np.sqrt(feature_map_x_diff*feature_map_x_diff + feature_map_y_diff*feature_map_y_diff)

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(feature_map[sample, :, :, idx].reshape(112, 112, 1), cmap = 'Reds')
    ax1.set_title('Feature map')
    ax1.axis('off')
    ax3 = fig.add_subplot(1, 4, 2)
    ax3.imshow(feature_map_x_diff, cmap = 'Reds')
    ax3.set_title('x_derivative')
    ax3.axis('off')
    ax4 = fig.add_subplot(1, 4, 3)
    ax4.imshow(feature_map_y_diff, cmap = 'Reds')
    ax4.set_title('y_derivative')
    ax4.axis('off')
    ax2 = fig.add_subplot(1, 4, 4)
    ax2.imshow(diff_sqrt, cmap = 'Reds')
    ax2.set_title('Combined')
    ax2.axis('off')
    plt.show()

def make_derivative():
    
    folders = ['1st', '2nd', '3rd', '4th', '5th']

    for n_folder in folders:
        
        path = 'C:\\anaconda3\\envs\\torch\machine_learning\\pytorch_exercise\\frequency\\frequency_cam\\{}_epoch\\'.format(n_folder)

        feature_map = np.transpose(np.load(path + 'first_map.npy'), (0, 2, 3, 1))

        feature_map_left = np.zeros((50, 112, 112, 64))
        feature_map_up = np.zeros((50, 112, 112, 64))

        feature_map_left[:, :111, :, :] = feature_map[:, 1: ,:, :]
        feature_map_up[:, :, :111, :] = feature_map[:, :, 1:, :]

        feature_map_x_diff = np.abs(feature_map[:, :, :, :].reshape(50, 112, 112, 64)-feature_map_left[:, :, :, :].reshape(50, 112, 112, 64))
        feature_map_y_diff = np.abs(feature_map[:, :, :, :].reshape(50, 112, 112, 64)-feature_map_up[:, :, :, :].reshape(50, 112, 112, 64))

        diff_sqrt = np.sqrt(feature_map_x_diff*feature_map_x_diff + feature_map_y_diff*feature_map_y_diff)

        np.save(path + 'derivative.npy', diff_sqrt)

def plot_gradcam():

    fix_randomness(42)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device(0)

    if not True:
        print('Run baseline model...')
        recover_model = recovered_net(baseline_layers, 'W', True).to(device)
    else :
        print('Run target model...')
        recover_model = recovered_net(conv_layers, 'W', True).to(device)
        recover_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\adaptive_1_123.pth', map_location="cuda:0"))


    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True,  transform=test_transform)

    test_loader = DataLoader(test_set, batch_size = 5, shuffle = False, num_workers=2)

    image, target = iter(test_loader).next()
    index = 4
    sample, label = image[index], target[index]
    sample, label = sample.to(device).unsqueeze(0), label.to(device).unsqueeze(0)

    cam = grad_cam(recover_model)

    ret_cam, _ = cam.get_label_cam(sample, label)
    print(sample.shape)
    sample_np = np.transpose(sample.detach().cpu().numpy(), (0, 2, 3, 1)).squeeze(0)
    cam_np = ret_cam.detach().view(224, 224, 1).numpy()
    print(cam_np.shape)

    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(sample_np)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(cam_np, cmap = 'jet')
    plt.show()


if __name__ == '__main__':
    #plot_activation_cam()
    #check_derivative()
    #make_derivative()
    plot_gradcam()

