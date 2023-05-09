# TEST DATA LOADERS
import sys
sys.path.append('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation')

import torch
import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
from general_utils import utils
from general_utils import MYParser
from dataloading import dataloaders
from dataloading import datasets
from torch.utils.data import DataLoader


config_file = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/test_exp_cpu.py'
config      = MYParser.MyParser(config_file).config_namespace

#---------- data loaders
split_dict = utils.load_split_dict(config.path_data, 
                                    config.path_splits, 
                                    config.fold_id, 
                                    config.train_data_name, 
                                    config.valid_data_name, 
                                    config.test_data_name)


# TEST dataloaders
test_loader = dataloaders.get_dataloader_single('train',
                                                split_dict['train'],
                                                config.path_data,
                                                config.batch_size,
                                                config.shuffle_train,
                                                config.num_workers,
                                                config)

#---------- test data augmentations
source_iter = iter(test_loader)
slice       = 10
name, adj_mtx, node_feats, adj_mtx_gt, node_feats_gt  = next(source_iter)
'''
if image.shape[2]>slice:
    plt.figure(1)
    plt.imshow(image[:,:,slice])
    plt.figure(4)
    plt.imshow(segm[:,:,slice])
    plt.figure(5)
    plt.imshow(mask[:,:,slice])
plt.show()
plt.close()
'''
# for data augmentation just plot the images
# for features you can re assemply the init image
# for connections