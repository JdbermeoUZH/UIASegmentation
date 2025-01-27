from typing import Optional

import torch
import random
import numpy as np
import torchio as tio
import torch.nn.functional as F
from dataloading import datasets
from torch.utils.data import DataLoader


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
    # TODO: turn this into a strategy pattern
    # get transforms and dataset class
    if type_of_loader == 'train':
        if config.use_patches == True:
            # dataloaders for graph networks
            transform = get_transform_train(config)
            dataset   = datasets.UIAGraph_Dataset(path_data, data, transform, config)
        elif config.use_patches == False:
            # dataloaders for unet network
            transform = get_transform_train_v2(config)
            dataset = datasets.UIA_Dataset(path_data, data, transform, config)

    elif type_of_loader == 'validation':
        if config.use_patches == True:
            transform = get_transform_valid(config)
            dataset   = datasets.UIAGraph_Dataset(path_data, data, transform, config)
        elif config.use_patches == False:
            transform = get_transform_valid(config)
            dataset = datasets.UIA_Dataset(path_data, data, transform, config)
    
    elif type_of_loader == 'test':
        if config.use_patches == True:
            transform = get_transform_test(config)
            dataset   = datasets.UIAGraph_Dataset(path_data, data, transform, config)
        elif config.use_patches == False:
            transform = get_transform_test(config)
            dataset   = datasets.UIA_Dataset(path_data, data, transform, config)
    else:
        print("Error: Wrong type of loader in get_loaders_single")
        raise NameError

    custom_loader = DataLoader(dataset,
                               batch_size  = batch_size,
                               shuffle     = shuffle,
                               num_workers = num_workers,
                               pin_memory  = True)
    return custom_loader


#---------- helper functions
def get_transform_train(config):
    transform_label  = get_label_transform(config.experiment_type)
    transforms_train = ComposeTransforms([ToTensor(),
                                          RandomFlip(config.transforms_probability),
                                          RandomAffine(config.transforms_probability),
                                          transform_label])
    return transforms_train

def get_transform_train_v2(config):
    transform_label  = get_label_transform(config.experiment_type)
    transforms_train = ComposeTransforms([ToTensor(),
                                          RandomFlip(config.transforms_probability),
                                          RandomAffine(config.transforms_probability),
                                          transform_label])
    return transforms_train

def get_transform_valid(config):
    transform_label  = get_label_transform(config.experiment_type)
    transforms_valid = ComposeTransforms([ToTensor(),
                                          transform_label])
    return transforms_valid

def get_transform_test(config):
    transform_label  = get_label_transform(config.experiment_type)
    transforms_test  = ComposeTransforms([ToTensor(),
                                          transform_label])
    return transforms_test

def get_label_transform(experiment_type):
    if experiment_type == 'binary_class':
        # Binary class in which the aneurysm is the only target
        transforms = ComposeTransforms([BinarizeSegmentation(specific_label=4)])
        return transforms

    elif experiment_type == 'binary_class_all_vessels':
        # Binary class in which all vessels are the target
        transforms = ComposeTransforms([BinarizeSegmentation(specific_label=4)])

    elif experiment_type == 'three_class':
        # hard code aneyrism type = label 4
        transforms = ComposeTransforms([CollapseLabels([i for i in range(1, 22) if i != 4]),
                                        MapLabel(target_label = 4, new_label = 2),
                                        OneHotEncode(num_classes = 3)])
        return transforms
    elif experiment_type == 'multi_class':
        transforms = ComposeTransforms([OneHotEncode(num_classes = 22)])
        return None
    else:
        return None


#---------- Transformations
class ComposeTransforms():
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, item):
        for t in self.transforms:
            if t != None:
                try:
                    item = t(item)
                except Exception as e:
                    raise Exception(f'Transformation Error: in {str(t)} with error {str(e)}')
        return item
    
class ToTensor():
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, item):
        imag = item['imag']
        mask = item['mask']
        segm = item['segm']

        imag = torch.from_numpy(imag)
        imag = imag.type(torch.float32)

        mask = torch.from_numpy(mask)
        mask = mask.type(torch.int16)

        segm = torch.from_numpy(segm)
        segm = segm.type(torch.int8)

        item['imag'] = imag
        item['mask'] = mask
        item['segm'] = segm
        return item

class Padding():
    def __init__(self, kernel):
        if len(kernel) ==3:
            self.div_xaxis = kernel[0]
            self.div_yaxis = kernel[1]
            self.div_zaxis = kernel[2]
        elif len(kernel) == 1:
            self.div_xaxis = kernel[0]
            self.div_yaxis = kernel[0]
            self.div_zaxis = kernel[0]
        else:
            print(f'Inside the Padding method, the variable {kernel} is not well defined')
            raise TypeError
    
    def __call__(self, item):
        imag = item['imag']
        mask = item['mask']
        segm = item['segm']
        
        assert imag.shape == mask.shape == segm.shape, "Inside Pad the dimensions don't fit"

        padx_left = 0 if imag.shape[0] % self.div_xaxis == 0 else (imag.shape[0] // self.div_xaxis + 1) * self.div_xaxis - imag.shape[0]
        pady_left = 0 if imag.shape[1] % self.div_yaxis == 0 else (imag.shape[1] // self.div_yaxis + 1) * self.div_yaxis - imag.shape[1]
        padz_left = 0 if imag.shape[2] % self.div_zaxis == 0 else (imag.shape[2] // self.div_zaxis + 1) * self.div_zaxis - imag.shape[2]

        padx_right = padx_left//2 if padx_left%2 == 0 else padx_left//2+1
        pady_right = pady_left//2 if pady_left%2 == 0 else pady_left//2+1
        padz_right = padz_left//2 if padz_left%2 == 0 else padz_left//2+1

        padx_left = padx_left//2
        pady_left = pady_left//2
        padz_left = padz_left//2

        imag = F.pad(imag, pad = (padz_left, padz_right, pady_left, pady_right, padx_left, padx_right))
        mask = F.pad(mask, pad = (padz_left, padz_right, pady_left, pady_right, padx_left, padx_right))
        segm = F.pad(segm, pad = (padz_left, padz_right, pady_left, pady_right, padx_left, padx_right))

        item['imag'] = imag
        item['mask'] = mask
        item['segm'] = segm
        
        return item
    
class RandomFlip():
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, item):
        random_probs = np.random.rand(3)
        imag = item['imag']
        mask = item['mask']
        segm = item['segm']

        if random_probs[0]>1-self.prob:
            imag = torch.flip(imag, dims = [0])
            mask = torch.flip(mask, dims = [0])
            segm = torch.flip(segm, dims = [0])
        if random_probs[1]>1-self.prob:
            imag = torch.flip(imag, dims = [1])
            mask = torch.flip(mask, dims = [1])
            segm = torch.flip(segm, dims = [1])
        if random_probs[2]>1-self.prob:
            imag = torch.flip(imag, dims = [2])
            mask = torch.flip(mask, dims = [2])
            segm = torch.flip(segm, dims = [2])

        item['imag'] = imag
        item['mask'] = mask
        item['segm'] = segm

        return item

class RandomRotate_90_180_270():
    def __init__(self, prob = 0.5):
        self.prob   = prob
        self.angles = [1,2,3] 
    
    def __call__(self, item):
        random_probs  = np.random.rand(3)
        random_angles = np.random.randint(0,3, size = 3)
        imag = item['imag']
        mask = item['mask']
        segm = item['segm']
        # rotate on xy-plane
        if random_probs[0] > 1-self.prob:
            imag = torch.rot90(imag, self.angles[random_angles[0]], [0,1])
            mask = torch.rot90(mask, self.angles[random_angles[0]], [0,1])
            segm = torch.rot90(segm, self.angles[random_angles[0]], [0,1])
        # rotate on yz-plane
        if random_probs[1] > 1-self.prob:
            imag = torch.rot90(imag, self.angles[random_angles[1]], [1,2])
            mask = torch.rot90(mask, self.angles[random_angles[1]], [1,2])
            segm = torch.rot90(segm, self.angles[random_angles[1]], [1,2])
        # rotate on zx-plane
        if random_probs[2] > 1-self.prob:
            imag = torch.rot90(imag, self.angles[random_angles[2]], [0,2])
            mask = torch.rot90(mask, self.angles[random_angles[2]], [0,2])
            segm = torch.rot90(segm, self.angles[random_angles[2]], [0,2])
    
        item['imag'] = imag
        item['mask'] = mask
        item['segm'] = segm

        return item

class RandomAffine():
    '''
    Adopted from 
    https://torchio.readthedocs.io/transforms/augmentation.html#randomaffine
    '''
    def __init__(self, prob = 0.5):
        self.prob        = prob
        self.scales      = (0.9, 1.2)
        self.degrees     = 15
        self.rand_affine = tio.transforms.RandomAffine(scales = self.scales, degrees = self.degrees) 
    
    def __call__(self, item):
        random_prob  = np.random.rand()
        imag         = item['imag']
        mask         = item['mask']
        segm         = item['segm']

        assert imag.ndim == mask.ndim == segm.ndim, "Inside Random Affine: dimensions mismatch"

        if random_prob > 1-self.prob:
            if imag.ndim == 3:
                imag = imag.unsqueeze(0)
                mask = mask.unsqueeze(0)
                segm = segm.unsqueeze(0)
            
            subject = tio.Subject(image1 = tio.ScalarImage(tensor = imag),
                                  image2 = tio.LabelMap(tensor = mask),
                                  image3 = tio.LabelMap(tensor = segm))
            subject = self.rand_affine(subject)
            imag = subject['image1'].data.squeeze(0)
            mask = subject['image2'].data.squeeze(0)
            segm = subject['image3'].data.squeeze(0)
        
        item['imag'] = imag
        item['mask'] = mask
        item['segm'] = segm

        return item
    
class RandomElastic():
    '''
    Adopted from 
    https://torchio.readthedocs.io/transforms/augmentation.html#randomaffine
    The specific transformation is slow as mentioned in the official documentation
    Better avoid it for big images.
    '''
    def __init__(self, prob = 0.1):
        self.prob                = prob
        self.num_control_points  = 7
        self.max_displacement    = 7.5
        self.locked_borders      = 2
        self.image_interpolation = 'linear'
        self.label_interpolation = 'nearest'
        self.randelastic         = tio.RandomElasticDeformation(num_control_points  = self.num_control_points,
                                                                max_displacement    = self.max_displacement,
                                                                locked_borders      = self.locked_borders,
                                                                image_interpolation = self.image_interpolation,
                                                                label_interpolation = self.label_interpolation)

    def __call__(self, item):
        random_prob  = np.random.rand()
        imag         = item['imag']
        mask         = item['mask']
        segm         = item['segm']

        assert imag.ndim == mask.ndim == segm.ndim, "Inside Random Affine: dimensions mismatch"

        if random_prob > 1-self.prob:
            if imag.ndim == 3:
                imag = imag.unsqueeze(0)
                mask = mask.unsqueeze(0)
                segm = segm.unsqueeze(0)
            
            subject = tio.Subject(image1 = tio.ScalarImage(tensor = imag),
                                  image2 = tio.LabelMap(tensor = mask),
                                  image3 = tio.LabelMap(tensor = segm))
            
            subject = self.randelastic(subject)
            imag = subject['image1'].data.squeeze(0)
            mask = subject['image2'].data.squeeze(0)
            segm = subject['image3'].data.squeeze(0)
        
        item['imag'] = imag
        item['mask'] = mask
        item['segm'] = segm

        return item

#---------- Label Transformations
class BinarizeSegmentation():
    """
    This creates an aneurysm only target
    """
    def __init__(self, specific_label: Optional[int] = None):
        self.specific_label = specific_label

    def __call__(self, item):
        segm         = item['segm']
        
        #--- only for testing
        item['segm_before_bin'] = segm
        #only for testing ---

        if self.specific_label is not None:
            segm      = torch.where(segm == self.specific_label, 1, 0)
        else:
            segm      = torch.where(segm > 0, 1, 0)

        item['segm'] = segm

        return item

class CollapseLabels():
    def __init__(self, target_labels):
        # target labels must be sorted
        self.master_label  = target_labels.pop(0)
        self.target_labels = target_labels
         

    def __call__(self, item):
        segm = item['segm']
        for tlabel in self.target_labels:
            segm = torch.where(segm == tlabel, self.master_label, segm)
        item['segm'] = segm
        return item

class MapLabel():
    def __init__(self, target_label, new_label):
        self.target_label = target_label
        self.new_label    = new_label
    
    def __call__(self, item):
        segm = item['segm']
        segm = torch.where(segm == self.target_label, self.new_label, segm)
        item['segm'] = segm
        return item

class OneHotEncode():
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, item):
        segm                       = item['segm']
        item['segm_before_onehot'] = segm

        segm = torch.nn.functional.one_hot(segm.long(), self.num_classes).to(torch.bool)
        segm = segm.permute(-1, *range(segm.dim() -1)).contiguous()
        item['segm'] = segm
        return item