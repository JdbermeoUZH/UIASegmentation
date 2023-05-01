import os
import h5py
import numpy as np
from torch.utils.data import Dataset


#---------- helper functions
def min_max_normalization(volume, percent_min, percent_max):
    
    obj_volume = volume[np.where(volume > 0)]
    
    min_value  = np.percentile(obj_volume, percent_min)
    max_value  = np.percentile(obj_volume, percent_max)
    
    obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    obj_volume[np.where(obj_volume > 1)] = 1
    obj_volume[np.where(obj_volume < 0)] = 0

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume

    return volume


def standardization(volume):
    mean   = np.mean(volume)
    var    = np.var(volume)
    volume = (volume - mean)/var
    return volume


#---------- datasets
class UIAGraph_Dataset(Dataset):
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
        self.experiment        = config.experiment_type
        self.normalization     = config.normalization
        self.norm_percent_max  = config.max_normalization_percent
        self.norm_percent_min  = config.min_normalization_percent

    def __len__(self):
        length = len(self.data)
        return length
    
    def __getitem__(self, idx):
        current_item = self.data[idx]
        current_name = current_item['name']
        # image has the intensities of the init tof image. Maybe without the skull => float32
        with h5py.File(current_item['imag'], 'r') as f: current_imag = f['data'][()]
        # mask is supposed to be number of components => uint16
        with h5py.File(current_item['mask'], 'r') as f: current_mask = f['data'][()]
        # segm is supposed to be with values in range 0-21 => uint8
        with h5py.File(current_item['segm'], 'r') as f: current_segm = f['data'][()]

        # normalize the init image
        if self.normalization == 'min_max':
            current_imag = min_max_normalization(current_imag, 
                                                 self.norm_percent_min,
                                                 self.norm_percent_max)
        elif self.normalization == 'standardization':
            current_imag = standardization(current_imag)
        
        current_imag = current_imag.astype(np.float32)
        current_mask = current_mask.astype(np.int16)
        current_segm = current_segm.astype(np.int8)

        assert current_imag.ndim == current_segm.ndim == current_mask.ndim, f'In the Dataset class mismatch in dimensions of images {current_name}'
        
        new_item = {'name': current_name, 'imag': current_imag, 'mask': current_mask, 'segm': current_segm}
        if self.transform != None:  new_item = self.transform(new_item)

        # compute graph's adjacency matrix and node features
        
        return new_item['name'], new_item['imag'], new_item['mask'], new_item['segm']