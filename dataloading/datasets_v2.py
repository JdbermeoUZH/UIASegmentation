import os
import h5py
import torch
import numpy as np
from general_utils import utils
from torch.utils.data import Dataset

#---------- datasets
class UIA_Dataset_v2(Dataset):
    def __init__(self, path_data, data, transform = None, config = None):

        if os.path.exists(path_data) == False:
            print(f'Path to data {path_data} doesnt exists... Exiting from Dataset init')
            raise FileNotFoundError
        
        if not isinstance(data, list) :
            print(f'data is type of {type(data)}... Exiting from Dataset init')
            raise TypeError
        
        if not (len(data) >0):
            print('data list is empty... Exiting from Dataset init')
            raise ValueError
        
        super().__init__()
        self.transform  = transform
        self.path_data  = path_data
        self.data       = data
        
        if config == None:
            print('Configuration is not given... Exiting from Dataset init')
            raise FileNotFoundError
        
        self.experiment = config.experiment_type

    def __len__(self):
        length = len(self.data)
        return length
    
    def __getitem__(self, idx):
        
        current_item = self.data[idx]
        current_name = current_item['name']
        
        # image has the intensities of the predicted segmented vessels.
        # The images have already been processed. So the values are in the range  0-1
        with h5py.File(current_item['imag'], 'r') as f: current_imag = f['data'][()]
        # The masks are the predicted masks with 1 or 0 => uint8
        with h5py.File(current_item['mask'], 'r') as f: current_mask = f['data'][()]
        # segm is supposed to be with values in range 0-21 => uint8
        with h5py.File(current_item['segm'], 'r') as f: current_segm = f['data'][()]
        
        current_imag = current_imag.astype(np.float32)
        current_mask = current_mask.astype(np.int8)
        current_segm = current_segm.astype(np.int8)

        assert current_imag.ndim == current_segm.ndim == current_mask.ndim, f'In the Dataset class, mismatch in dimensions of images {current_name}'
        assert current_imag.shape == current_segm.shape == current_mask.shape, f'In the Dataset class, mismatch in shapes of images {current_name}'
        
        new_item = {'name': current_name, 
                    'imag': current_imag, 
                    'mask': current_mask, 
                    'segm': current_segm}
        
        if self.transform != None:  
            new_item = self.transform(new_item)
        
        return current_name, new_item['imag'].unsqueeze(0), new_item['mask'].unsqueeze(0), new_item['segm'].unsqueeze(0)
        