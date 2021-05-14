import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from machine_learning.pytorch_exercise.frequency.vgg_recover import recovered_net
from machine_learning.pytorch_exercise.cnn_cifar10.random_seed import fix_randomness
from machine_learning.pytorch_exercise.cnn_cifar10.model import train_save_model
from machine_learning.pytorch_exercise.cnn_cifar10.model import custom_dataset
from machine_learning.pytorch_exercise.cnn_cifar10.cam import grad_cam
from machine_learning.pytorch_exercise.cnn_cifar10.about_image import tensor_to_numpy
from machine_learning.pytorch_exercise.cnn_cifar10.about_image import inverse_normalize
import time
from PIL import Image
from sklearn.metrics import confusion_matrix
from machine_learning.pytorch_exercise.cnn_cifar10.SaveResult.save_confusion_matrix import plot_confusion_matrix
    

def drive():

    fix_randomness(123)
    
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    device = torch.device('cuda')


    print('Run baseline model...')
    baseline_model = recovered_net(baseline_layers, 'W', True).to(device)
    baseline_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\baseline_123.pth', map_location="cuda:0"))

    print('Run target model...')
    recover_model = recovered_net(conv_layers, 'W', True).to(device)
    recover_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\adaptive_0_123.pth', map_location="cuda:0"))


    data = np.load('C:\\anaconda3\envs\\torch\machine_learning\\subset_zero\\sample_ff.npy')
    target = np.load('C:\\anaconda3\envs\\torch\machine_learning\\subset_zero\\label_ff.npy')

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_set = custom_dataset.CustomDataset(data, target, transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0) 
        
    baseline_cam = grad_cam.grad_cam(baseline_model)
    recover_cam = grad_cam.grad_cam(recover_model)

    baseline_model.eval()
    recover_model.eval()

    class_map = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    rows, cols = 2, 4
    fontsize = 24

    for img_idx, (sample, label) in enumerate(test_loader):

        path = 'C:\\anaconda3\envs\\torch\machine_learning\\cam_ff\\cam_zero\\{}'.format(img_idx)

        sample, label = sample.to(device), label.to(device)

        recover_ret, recover_pred = recover_cam.get_cam(sample, label)
        recover_ret, recover_pred = recover_ret.detach().cpu(), recover_pred.detach().cpu()
        baseline_ret, baseline_pred = baseline_cam.get_cam(sample, label)
        baseline_ret, baseline_pred = baseline_ret.detach().cpu(), baseline_pred.detach().cpu()

        recover_false_ret, recover_false_pred = recover_cam.get_label_cam(sample, baseline_pred)
        recover_false_ret, recover_false_pred = recover_false_ret.detach().cpu(), recover_false_pred.detach().cpu()
        baseline_false_ret, baseline_false_pred = baseline_cam.get_label_cam(sample, recover_pred)
        baseline_false_ret, baseline_false_pred = baseline_false_ret.detach().cpu(), baseline_false_pred.detach().cpu()

        recover_true_ret, recover_true_pred = recover_cam.get_label_cam(sample, label)
        recover_true_ret, recover_true_pred = recover_true_ret.detach().cpu(), recover_true_pred.detach().cpu()
        baseline_true_ret, baseline_true_pred = baseline_cam.get_label_cam(sample,label)
        baseline_true_ret, baseline_true_pred = baseline_true_ret.detach().cpu(), baseline_true_pred.detach().cpu()

        sample = sample.detach().cpu()

        sample_denorm = inverse_normalize(sample, mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010), batch = True)


        baseline_ret = tensor_to_numpy(baseline_ret.unsqueeze(0), False)
        recover_ret = tensor_to_numpy(recover_ret.unsqueeze(0), False)


        sample_denorm = tensor_to_numpy(sample_denorm, True).squeeze(0)


        if (label.item() == baseline_pred.item()) or (label.item() == recover_pred.item()): 
            print('Error Case.')
            continue

        fig = plt.figure(figsize = (16, 8))

        # Input Image
        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(sample_denorm)
        ax1.set_title(class_map[int(label)], fontsize = fontsize)
        ax1.axis('off')
        # Baseline Pred CAM
        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(sample_denorm)
        ax2.imshow(baseline_ret, cmap = 'jet', alpha = 0.4)
        ax2.set_title('baseline ({})'.format(class_map[int(baseline_pred)]), fontsize = fontsize)
        ax2.axis('off')
        # Baseline False CAM
        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(sample_denorm)
        ax3.imshow(baseline_false_ret, cmap = 'jet', alpha = 0.4)
        ax3.set_title(class_map[int(baseline_false_pred)], fontsize = fontsize)
        ax3.axis('off')
        # Baseline True CAM
        ax4 = fig.add_subplot(rows, cols, 4)
        ax4.imshow(sample_denorm)
        ax4.imshow(baseline_true_ret, cmap = 'jet', alpha = 0.4)
        ax4.set_title(class_map[int(baseline_true_pred)], fontsize = fontsize)
        ax4.axis('off')
        # Input Image
        ax5 = fig.add_subplot(rows, cols, 5)
        ax5.imshow(sample_denorm)
        ax5.set_title(class_map[int(label)], fontsize = fontsize)
        ax5.axis('off')
        # Target False CAM
        ax6 = fig.add_subplot(rows, cols, 6)
        ax6.imshow(sample_denorm)
        ax6.imshow(recover_false_ret, cmap = 'jet', alpha = 0.4)
        ax6.set_title('recover ({})'.format(class_map[int(recover_false_pred)]), fontsize = fontsize)
        ax6.axis('off')
        # Target Pred CAM
        ax7 = fig.add_subplot(rows, cols, 7)
        ax7.imshow(sample_denorm)
        ax7.imshow(recover_ret, cmap = 'jet', alpha = 0.4)
        ax7.set_title(class_map[int(recover_pred)], fontsize = fontsize)
        ax7.axis('off')
        # Target True CAM
        ax8 = fig.add_subplot(rows, cols, 8)
        ax8.imshow(sample_denorm)
        ax8.imshow(recover_true_ret, cmap = 'jet', alpha = 0.4)
        ax8.set_title(class_map[int(recover_true_pred)], fontsize = fontsize)
        ax8.axis('off')

        plt.savefig(path)
        #plt.show()
        print(img_idx, ' / ', len(test_loader), ' image saved.')
        plt.close()

def test():

    fix_randomness(123)

    device = torch.device(0)

    torch.cuda.empty_cache()

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download = True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle = False, num_workers=0)

    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    # print('Run baseline model...')
    # baseline_model = recovered_net(baseline_layers, 'W', True).to(device)
    # baseline_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\baseline_123.pth', map_location="cuda:0"))

    print('Run target model...')
    recover_model = recovered_net(conv_layers, 'W', True).to(device)
    recover_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\adaptive_123.pth', map_location="cuda:0"))

    recover_model.eval()

    recover_ret = torch.empty((10000, ))

    for idx, (sample, label) in enumerate(test_loader):
        sample, label = sample.to(device), label.to(device)

        output = recover_model.forward(sample)
        _, pred = torch.max(output, dim = 1)

        recover_ret[5*idx:5*(idx+1)] = pred

        print(idx+1)

    np.save('./half_ret.npy', recover_ret)    


def make_dataset():
    fix_randomness(123)

    device = torch.device(0)

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, batch_size=10000, shuffle = False, num_workers=0)

    sample, label = iter(test_loader).next()
    
    path = 'C:\\anaconda3\envs\\torch\machine_learning\\subset_zero\\'

    ft = np.load(path + 'ft.npy')
    ff = np.load(path + 'ff.npy')
    tf = np.load(path + 'tf.npy')

    sample_ft, label_ft = sample[ft], label[ft]
    print('ft : ', sample_ft.shape, label_ft.shape, ft.shape)

    sample_ff, label_ff = sample[ff], label[ff]
    print('ff : ', sample_ff.shape, label_ff.shape, ff.shape)

    sample_tf, label_tf = sample[tf], label[tf]
    print('tf : ', sample_tf.shape, label_tf.shape, tf.shape)


    np.save(path + 'sample_ft.npy', sample_ft)
    np.save(path + 'label_ft.npy', label_ft)
    np.save(path + 'sample_ff.npy', sample_ff)
    np.save(path + 'label_ff.npy', label_ff)
    np.save(path + 'sample_tf.npy', sample_tf)
    np.save(path + 'label_tf.npy', label_tf)

    print('dataset is saved')

def test2():
    fix_randomness(123)

    device = torch.device(0)

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data = np.load('./sample_wrong.npy')
    target = np.load('./label_wrong.npy')

    print(data.shape, target.shape)

    test_set = custom_dataset.CustomDataset(data, target, test_transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle = False, num_workers=0)

    baseline_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    conv_layers = [64, 'R', 128, 'R', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    print('Run target model...')
    recover_model = recovered_net(conv_layers, 'W', True).to(device)
    recover_model.load_state_dict(torch.load('C:\\anaconda3\envs\\torch\machine_learning\pytorch_exercise\\frequency\data\\adaptive_123.pth', map_location="cuda:0"))
    
    recover_model.eval()

    for sample, label in test_loader:
        sample, label = sample.to(device), label.int().to(device)

        output = recover_model.forward(sample)
        _, pred = torch.max(output, dim = 1)

        ld, pd = label.data, pred.data

        for (l, p) in zip(ld, pd):
            if l == p:
                print('Error Case')
            else:
                print('----')

def find():
    fix_randomness(123)

    baseline = np.load('./baseline_wrong.npy')
    one = np.load('./one_wrong.npy')
    half = np.load('./half_wrong.npy')
    zero = np.load('./zero_wrong.npy')

    path = './subset_zero\\'

    ft = np.setdiff1d(baseline, zero)
    tf = np.setdiff1d(zero, baseline)
    ff = np.intersect1d(baseline, zero)

    np.save(path + 'ft.npy', ft)
    np.save(path + 'tf.npy', tf)
    np.save(path + 'ff.npy', ff)

def get_confusion_matrix():

    fix_randomness(123)

    device = torch.device(0)

    test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    test_set = torchvision.datasets.CIFAR10('./data', train = False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=10000, shuffle = False, num_workers=0)

    sample, label = iter(test_loader).next()

    image = sample[17]
    image_np = np.transpose(image.numpy(), (1, 2, 0))

    red = image_np[:, :, 0].reshape(224, 224, 1)
    green = image_np[:, :, 1].reshape(224, 224, 1)
    blue = image_np[:, :, 2].reshape(224, 224, 1)

    fig = plt.figure(figsize = (16, 4))
    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(image_np)
    ax1 = fig.add_subplot(1, 4, 2)
    ax1.imshow(red, cmap = 'binary')
    ax1.set_title('red')
    ax2 = fig.add_subplot(1, 4, 3)
    ax2.imshow(green, cmap = 'binary')
    ax2.set_title('green')
    ax3 = fig.add_subplot(1, 4, 4)
    ax3.imshow(blue, cmap = 'binary')
    ax3.set_title('blue')
    plt.show()

if __name__ == '__main__':
    #drive()
    #find()
    #make_dataset()
    #test2()
    get_cam()