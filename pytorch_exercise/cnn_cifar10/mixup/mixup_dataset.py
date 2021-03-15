import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random_seed
import torchvision
from torchvision import transforms


def get_dataset(origin_dataset, recon_ratio, src_ratio, mixup_class, normalize) :
    # mixup_class : tuple data which means what class is mixed up
    loader = DataLoader(origin_dataset, batch_size = len(origin_dataset), shuffle = False, num_workers = 4)
    # treat all dataset, make no batch and set shuffle False because we put randomness later
    # we consider origin_dataset is Tensor type data, balanced dataset and we mixup images by one-to-one following indices
    image, label = iter(loader).next()
    print('image denormalizing')

    # undo normalization
    mean = torch.tensor(normalize[0], dtype = torch.float64)
    std = torch.tensor(normalize[1], dtype = torch.float64)
    unnormalizing = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    image = unnormalizing(image)
    search_idx = []


    for mixup in mixup_class :
        src_class, noise_class = mixup[0], mixup[1]
        # get src, noise indices in dataset
        src_idx = (label==src_class).nonzero().view(-1)
        noise_idx = (label==noise_class).nonzero().view(-1)

        n_src = src_idx.size()[0]
        n_noise = noise_idx.size()[0]
        n_mixup = int(n_src*recon_ratio)
        
        # n_src 중에서 recon_ratio 만큼의 indices를 random하게 extract
        src_apply_idx = torch.LongTensor(np.random.choice(n_src, n_mixup))
        noise_apply_idx = torch.LongTensor(np.random.choice(n_noise, n_mixup))

        # extract mixup indices by randomly, 실제 mixup이 적용되는 idx
        src_selected = src_idx[src_apply_idx]
        noise_selected = noise_idx[noise_apply_idx]

        # extract source, noise image
        src_image = image[src_selected]
        noise_image = image[noise_selected]

        # make mixup image and replace source class's original image
        mixup_image = (src_image*src_ratio) + (noise_image*(1-src_ratio))

        # mixed up image's indices in each source class, list's indices mean source class's number 
        search_idx.append(src_selected)        

        for index, replace_idx in enumerate(src_selected) :
            image[replace_idx] = mixup_image[index]
        print('add noise of class.{0} to class.{1}'.format(noise_class, src_class))

    # re-normalization
    normalizing = transforms.Normalize(mean = normalize[0], std = normalize[1])
    image = normalizing(image)

    ret_dataset = torch.utils.data.TensorDataset(image, label)

    return ret_dataset, search_idx


def get_dataset_added(origin_dataset, recon_ratio, src_ratio, mixup_class, normalize) :
    # mixup_class : tuple data which means what class is mixed up
    loader = DataLoader(origin_dataset, batch_size = len(origin_dataset), shuffle = False, num_workers = 4)
    # treat all dataset, make no batch and set shuffle False because we put randomness later
    # we consider origin_dataset is Tensor type data, balanced dataset and we mixup images by one-to-one following indices
    image, label = iter(loader).next()
    print('image denormalizing')
    # undo normalization
    mean = torch.tensor(normalize[0], dtype = torch.float64)
    std = torch.tensor(normalize[1], dtype = torch.float64)
    unnormalizing = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    image = unnormalizing(image)
    search_idx = []
    ret_image, ret_label = image, label


    for mixup in mixup_class :
        src_class, noise_class = mixup[0], mixup[1]
        # get src, noise indices in dataset
        src_idx = (label==src_class).nonzero().view(-1)
        noise_idx = (label==noise_class).nonzero().view(-1)

        n_src = src_idx.size()[0]
        n_noise = noise_idx.size()[0]
        n_mixup = int(n_src*recon_ratio)

        # n_src 중에서 recon_ratio 만큼의 indices를 random하게 extract
        src_apply_idx = torch.LongTensor(np.random.choice(n_src, n_mixup))
        noise_apply_idx = torch.LongTensor(np.random.choice(n_noise, n_mixup))

        # extract mixup indices by randomly, 실제 mixup이 적용되는 idx
        src_selected = src_idx[src_apply_idx]
        noise_selected = noise_idx[noise_apply_idx]

        # extract source, noise image
        src_image = image[src_selected]
        noise_image = image[noise_selected]

        # make mixup image and replace source class's original image
        mixup_image = (src_image*src_ratio) + (noise_image*(1-src_ratio))

        # mixed up image's indices in each source class, list's indices mean source class's number 
        search_idx.append(src_selected)        

        # make mixup_image's label
        mixup_label = torch.full([n_mixup], src_class, dtype=torch.int64)

        # concatenate origin_dataset and mixup_dataset
        ret_image = torch.cat([ret_image, mixup_image], dim = 0)
        ret_label = torch.cat([ret_label, mixup_label], dim = 0)
        print('add noise of class.{0} to class.{1}'.format(noise_class, src_class))
    
    shuffle_idx = np.random.choice(ret_image.size()[0], ret_image.size()[0])
    ret_image, ret_label = ret_image[shuffle_idx], ret_label[shuffle_idx]

    # re-normalization
    normalizing = transforms.Normalize(mean = normalize[0], std = normalize[1])
    image = normalizing(image)
    
    ret_dataset = torch.utils.data.TensorDataset(ret_image, ret_label)

    return ret_dataset, search_idx


