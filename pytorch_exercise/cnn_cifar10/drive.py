import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import random
import random_seed
from model import train_save_model
from model import cnn, vgg16
from mixup import get_confusion_matrix
from mixup import mixup_dataset

# Apply random seed to all randomness

n_training = 9

def drive() :
    print('Training.')
    random_seed.make_random(42)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5),
            #transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])

    # Download test dataset, load data
    train_data = torchvision.datasets.CIFAR100(root= './data', download = True, train = True, transform = train_transform)
    test_data = torchvision.datasets.CIFAR100(root = './data', download = True, train = False, transform = valid_transform)

    recon_ratio = 0.5
    src_ratio = 0.70
    mixup_list = ((3, 5), (3, 6))

    if n_training > 100 :

        train_data_mixup, _ = mixup_dataset.get_dataset(train_data, recon_ratio, src_ratio, mixup_list, ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        train_loader = DataLoader(train_data_mixup, batch_size=50, shuffle = True, num_workers = 4)
        print('mixup dataset constructed!')

        with open('pytorch_exercise/cnn_cifar10/model_saved/tracking.txt', 'a') as f:
            f.write('Training_{0} \nrecon_ratio : {1} \nsrc_ratio : {2} \nmixup_list : {3}\n\n\n'.format(n_training, recon_ratio, src_ratio, mixup_list))
    else :
        train_loader = DataLoader(train_data, batch_size=50, shuffle = True, num_workers = 4)

        with open('pytorch_exercise/cnn_cifar10/model_saved/tracking.txt', 'a') as f:
            f.write('Training_{0} \nrecon_ratio : {1} \nsrc_ratio : {1} \nmixup_list : {1}\n\n\n'.format(n_training, 'None'))

    test_loader = DataLoader(test_data, batch_size=50, shuffle = False, num_workers = 4)

        
    load_path = 'pytorch_exercise/cnn_cifar10/model_saved/0316_{}.pth'.format(n_training-1)
    save_path = 'pytorch_exercise/cnn_cifar10/model_saved/0316_{}.pth'.format(n_training)

    # parameter is learning rate
    model = vgg16.vgg_net()

    if n_training > 100 :
        model.load_state_dict(torch.load(load_path))

    # training and save model
    train_eval_history = train_save_model.save_model(model, 50, train_loader, test_loader, save_path)
    
    # plot training, evaluation plot
    plot_save_path = 'pytorch_exercise/cnn_cifar10/model_saved/0316_plot_{}.jpg'.format(n_training)
    train_save_model.save_plot(train_eval_history, plot_save_path)
    print('Training Complete.')

    return test_loader

def evaluation(test_loader) :
    print('Evaluation.')
    random_seed.make_random(42)


    load_path = 'pytorch_exercise/cnn_cifar10/model_saved/0316_{}.pth'.format(n_training)

    model = vgg16.vgg_net()
    model.load_state_dict(torch.load(load_path))

    cf = get_confusion_matrix.get_cf_matrix(model, test_loader)
    plt.matshow(cf, cmap = 'binary')
    plt.savefig('pytorch_exercise/cnn_cifar10/model_saved/0316_cf_{}.jpg'.format(n_training))
    plt.show()
    

if __name__ == '__main__':
    test_loader = drive()
    evaluation(test_loader = test_loader)
