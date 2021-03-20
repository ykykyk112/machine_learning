import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
import random
import random_seed
from model import train_save_model
from model import cnn, vgg16, resnet
from mixup import get_confusion_matrix
from mixup import mixup_dataset
from SaveResult import save_confusion_matrix

n_training = 1

def train_model() :
    print('Empty cache')
    torch.cuda.empty_cache()
    print('Training.')
    random_seed.make_random(42)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(112, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
            transforms.Resize(112),
            transforms.ToTensor(),
            normalize,
        ])


    # Download test dataset, load data
    train_data = torchvision.datasets.CIFAR10(root= './data', download = True, train = True, transform = train_transform)
    test_data = torchvision.datasets.CIFAR10(root = './data', download = True, train = False, transform = valid_transform)



    train_loader = DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=50, shuffle = False, num_workers = 4)

        
    load_path = 'pytorch_exercise/cnn_cifar10/model_saved/0320_{}.pth'.format(n_training-1)
    save_path = 'pytorch_exercise/cnn_cifar10/model_saved/0320_{}.pth'.format(n_training)

    # parameter is learning rate
    model = vgg16.vgg_cam()

    if n_training > 100 :
        model.load_state_dict(torch.load(load_path))

    # training and save model
    train_eval_history = train_save_model.save_model(model, 40, train_loader, test_loader, save_path, cam_mode = True)
    
    # plot training, evaluation plot
    plot_save_path = 'pytorch_exercise/cnn_cifar10/model_saved/0320_plot_{}.jpg'.format(n_training)
    train_save_model.save_plot(train_eval_history, plot_save_path)
    print('Training Complete.')

    return test_loader


def evaluation(test_loader) :
    print('Evaluation.')
    random_seed.make_random(42)
    

    load_path = 'pytorch_exercise/cnn_cifar10/model_saved/0319_{}.pth'.format(n_training)

    model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(load_path))

    cf = get_confusion_matrix.get_cf_matrix(model, test_loader)
    save_confusion_matrix.plot_confusion_matrix(cf, ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    

if __name__ == '__main__':
    test_loader = train_model()
    #evaluation(test_loader = test_loader)
