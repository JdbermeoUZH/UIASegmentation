import os
import cc3d
import h5py
import torch
import numpy as np
from general_utils import utils
from torch.utils.data import Dataset


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
        self.patch_size        = config.graph_patch_size
        self.connectivity      = config.graph_connectivity

    def __len__(self):
        length = len(self.data)
        return length
    
    def __getitem__(self, idx):
        
        current_item = self.data[idx]
        current_name = current_item['name']
        
        # image has the intensities of the init tof image. Maybe without the skull => float32
        with h5py.File(current_item['imag'], 'r') as f: current_imag = f['data'][()]
        # mask contains the components => uint16
        with h5py.File(current_item['mask'], 'r') as f: current_mask = f['data'][()]
        # segm is supposed to be with values in range 0-21 => uint8
        with h5py.File(current_item['segm'], 'r') as f: current_segm = f['data'][()]

        # normalize the init image
        if self.normalization == 'min_max':
            current_imag = utils.min_max_normalization(current_imag, 
                                                       self.norm_percent_min,
                                                       self.norm_percent_max)
        elif self.normalization == 'standardization':
            current_imag = utils.standardization(current_imag)
        
        current_imag = current_imag.astype(np.float32)
        current_mask = current_mask.astype(np.int16)
        current_segm = current_segm.astype(np.int8)

        assert current_imag.ndim == current_segm.ndim == current_mask.ndim, f'In the Dataset class mismatch in dimensions of images {current_name}'
        
        new_item = {'name': current_name, 
                    'imag': current_imag, 
                    'mask': current_mask, 
                    'segm': current_segm}
        
        if self.transform != None:  
            new_item = self.transform(new_item)
        
        # compute graph's adjacency matrix and node features based on coarse mask
        #adj_mtx    = utils.get_adjacency_matrix(new_item['mask'], 
        #                                        self.patch_size, 
        #                                        self.connectivity)
        #adj_mtx = adj_mtx.to_sparse_coo()
        #adj_mtx = adj_mtx.indices()

        node_feats = utils.get_nodes_features(new_item['imag'], 
                                              self.patch_size)
        
        # compute graph's adjacency matrix and node features based on groundtruth mask
        #segm_np = new_item['segm'].numpy()
        #if self.experiment != 'binary_class':   segm_np = np.where(segm_np > 0, 1, 0)
        
        #segm_np1        = cc3d.connected_components(segm_np, 
        #                                            connectivity = self.connectivity)
        #segm_con_tensor = torch.from_numpy(segm_np1.astype('int16'))
        #del segm_np, segm_np1 #remove unnecessary variables

        #adj_mtx_gt    = utils.get_adjacency_matrix(segm_con_tensor,
        #                                           self.patch_size,
        #                                           self.connectivity)
        #del segm_con_tensor #remove unnecessary variables
        #adj_mtx_gt = adj_mtx_gt.to_sparse_coo()
        #adj_mtx_gt = adj_mtx_gt.indices()

        node_feats_gt = utils.get_nodes_features(new_item['segm'],
                                                 self.patch_size)
        node_feats    = node_feats.unsqueeze(1)
        node_feats_gt = node_feats_gt.unsqueeze(1)
        del new_item
        return torch.tensor([0]), node_feats, torch.tensor([0]), node_feats_gt