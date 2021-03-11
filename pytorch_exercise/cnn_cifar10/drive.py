import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import random_seed
from model import train_save_model
from model import cnn
from mixup import get_confusion_matrix
from mixup import mixup_dataset

# Apply random seed to all randomness

def drive() :
    print('Training.')
    random_seed.make_random(42)
    transform = transforms.Compose([    transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  ])
    # Download test dataset, load data
    train_data = torchvision.datasets.CIFAR10(root= './data', download = True, train = True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root = './data', download = True, train = False, transform = transform)

    #train_data_mixup, _ = mixup_dataset.get_dataset(train_data, 0.2, 0.7, ((2, 3), (3, 4), (5, 3)), True)
    print('mixup dataset constructed!')
    #train_loader = DataLoader(train_data_mixup, batch_size=50, shuffle = True, num_workers = 4)
    #test_loader = DataLoader(test_data, batch_size=50, shuffle = False, num_workers = 4)

    load_path = './model_saved/0311_4.pth'
    save_path = './model_saved/0311_5.pth'

    model = cnn.conv_net()
    #model.load_state_dict(torch.load(load_path))

    #train_save_model.save_model(model, 20, train_loader, test_loader, save_path)


def evaluation() :
    print('Evaluation.')
    random_seed.make_random(42)
    transform = transforms.Compose([    transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  ])
    # Download test dataset, load data
    test_data = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform)

    test_loader = DataLoader(test_data, batch_size=50, shuffle = False, num_workers = 4)

    load_path = './model_saved/0311_4.pth'

    model = cnn.conv_net()
    model.load_state_dict(torch.load(load_path))

    cf = get_confusion_matrix.get_cf_removed(model, test_loader)
    plt.matshow(cf, cmap = 'binary')
    plt.show()

if __name__ == '__main__':
    drive()
    #evaluation()
