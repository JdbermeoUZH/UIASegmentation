import torch
import random
import numpy as np
import torchio as tio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloading import dataloaders as dl
from dataloading import datasets_v2 as datasets

def get_dataloaders_all(config, split_dict):

    train_loader = get_dataloader_single('train',
                                         split_dict['train'],
                                         config.path_data,
                                         config.batch_size,
                                         config.shuffle_train,
                                         config.num_workers,
                                         config
                                         )
    valid_loader = get_dataloader_single('validation',
                                         split_dict['valid'],
                                         config.path_data,
                                         config.batch_size_val,
                                         config.shuffle_validation,
                                         config.num_workers,
                                         config)
    test_loader  = get_dataloader_single('test',
                                         split_dict['test'],
                                         config.path_data,
                                         config.batch_size_test,
                                         config.shuffle_test,
                                         config.num_workers,
                                         config
                                         )

    return train_loader, valid_loader, test_loader

def get_dataloader_single(type_of_loader, 
                          data, 
                          path_data, 
                          batch_size, 
                          shuffle, 
                          num_workers,
                          config):
    # get transforms and dataset class
    if type_of_loader == 'train':
        transform = dl.get_transform_train_v2(config)
    elif type_of_loader == 'validation':
        transform = dl.get_transform_valid(config)
    elif type_of_loader == 'test':
        transform = dl.get_transform_test(config)
    else:
        print("Error: Wrong type of loader in get_loaders_single")
        raise NameError
    
    dataset       = datasets.UIA_Dataset_v2(path_data, data, transform, config)
    custom_loader = DataLoader(dataset,
                               batch_size  = batch_size,
                               shuffle     = shuffle,
                               num_workers = num_workers,
                               pin_memory  = True)
    return custom_loader