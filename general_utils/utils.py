import os
import torch
import numpy as np
import pandas as pd


#---------- helper functions for preprocessing
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


#---------- helper functions for graph creation
def get_patches(volume, patch_size):
    stride  = patch_size # no overlaps
    patches = volume.unfold(0, patch_size[0], stride[0])\
                    .unfold(1, patch_size[1], stride[1])\
                    .unfold(2, patch_size[2], stride[2])
    patches = patches.contiguous()
    return patches

def get_nodes_features(volume, patch_size):
    patches = get_patches(volume, patch_size)
    # for ease traverse first x-axis then y-axis and then z-axis
    # id of a node is x + y*n_x + z*(n_x*n_y)
    return patches.permute(2,1,0,3,4,5)\
                  .reshape(-1,patch_size[0],patch_size[1],patch_size[2]) 

def find_node_id(x,y,z, patches_size):
    node_id = x + y*patches_size[0] + z*(patches_size[0]*patches_size[1])
    return node_id

def check_face(adj_mtx, x, y, z, neigh_x, neigh_y, neigh_z, face_target, face_neigh, patches_size):
    node_id_target = find_node_id(x, y, z, patches_size) 
    node_id_neigh  = find_node_id(neigh_x, neigh_y, neigh_z, patches_size)
    if adj_mtx[node_id_target, node_id_neigh] == 0: 
        common_values = face_target[torch.where(face_target == face_neigh)]
        if torch.any(common_values): #doesn't count 0 which is the background
            adj_mtx[node_id_target, node_id_neigh] = 1
            adj_mtx[node_id_neigh, node_id_target] = 1

def check_edge(adj_mtx, x, y, z, neigh_x, neigh_y, neigh_z, edge_target, edge_neigh, patches_size):
    node_id_target = find_node_id(x, y, z, patches_size) 
    node_id_neigh  = find_node_id(neigh_x, neigh_y, neigh_z, patches_size)
    if adj_mtx[node_id_target, node_id_neigh] == 0:
        common_values = edge_target[torch.where(edge_target == edge_neigh)]
        if torch.any(common_values):
            adj_mtx[node_id_target, node_id_neigh] = 1
            adj_mtx[node_id_neigh, node_id_target] = 1

def check_corner(adj_mtx, x, y, z, neigh_x, neigh_y, neigh_z, corner_target, corner_neigh, patches_size):
    node_id_target = find_node_id(x, y, z, patches_size)
    node_id_neigh  = find_node_id(neigh_x, neigh_y,neigh_z, patches_size)
    if adj_mtx[node_id_target, node_id_neigh] == 0:
        if corner_target == corner_neigh and corner_target != 0:
            adj_mtx[node_id_target, node_id_neigh] = 1
            adj_mtx[node_id_neigh, node_id_target] = 1

def find_adjacent_nodes6(adj_mtx, m_patches, x, y, z):
    size =  list(m_patches.shape)
    for z_neigh in range(max(0, z-1), 1+min(m_patches.shape[2]-1, z+1)):
        for y_neigh in range(max(0, y-1), 1+min(m_patches.shape[1]-1, y+1)):
            for x_neigh in range(max(0, x-1), 1+min(m_patches.shape[0]-1, x+1)):
                if z_neigh == z and y_neigh == y and x_neigh == x: 
                    continue
                if x_neigh == x and y_neigh == y:
                    if z_neigh > z:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,:,-1], m_patches[x_neigh, y_neigh, z_neigh,:,:,0], size)
                    elif z_neigh < z:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,:,0], m_patches[x_neigh, y_neigh, z_neigh,:,:,-1], size)
                elif x_neigh == x and z_neigh == z:
                    if y_neigh > y:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,-1,:], m_patches[x_neigh, y_neigh, z_neigh,:,0,:], size)
                    elif y_neigh < y:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,0,:], m_patches[x_neigh, y_neigh, z_neigh,:,-1,:], size)
                elif y_neigh == y and z_neigh == z:
                    if x_neigh > x:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,:,:], m_patches[x_neigh, y_neigh, z_neigh,0,:,:], size)
                    else:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,:,:], m_patches[x_neigh, y_neigh, z_neigh,-1,:,:], size)

def find_adjacent_nodes18(adj_mtx, m_patches, x, y, z):
    size =  list(m_patches.shape)
    for z_neigh in range(max(0, z-1), 1+min(m_patches.shape[2]-1, z+1)):
        for y_neigh in range(max(0, y-1), 1+min(m_patches.shape[1]-1, y+1)):
            for x_neigh in range(max(0, x-1), 1+min(m_patches.shape[0]-1, x+1)):         
                if z_neigh == z and y_neigh == y and x_neigh == x: 
                    continue
                # check faces
                if x_neigh == x and y_neigh == y:
                    if z_neigh > z:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,:,-1], m_patches[x_neigh, y_neigh, z_neigh,:,:,0], size)
                    elif z_neigh < z:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,:,0], m_patches[x_neigh, y_neigh, z_neigh,:,:,-1], size)
                elif x_neigh == x and z_neigh == z:
                    if y_neigh > y:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,-1,:], m_patches[x_neigh, y_neigh, z_neigh,:,0,:], size)
                    elif y_neigh < y:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,0,:], m_patches[x_neigh, y_neigh, z_neigh,:,-1,:], size)
                elif y_neigh == y and z_neigh == z:
                    if x_neigh > x:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,:,:], m_patches[x_neigh, y_neigh, z_neigh,0,:,:], size)
                    else:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,:,:], m_patches[x_neigh, y_neigh, z_neigh,-1,:,:], size)
                #check edges
                elif x_neigh == x:
                    if   z_neigh > z and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,-1,-1], m_patches[x_neigh, y_neigh, z_neigh,:,0,0], size)
                    elif z_neigh > z and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,0,-1], m_patches[x_neigh, y_neigh, z_neigh,:,-1,0], size)
                    elif z_neigh < z and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,-1,0], m_patches[x_neigh, y_neigh, z_neigh,:,0,-1], size)
                    elif z_neigh < z and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,0,0], m_patches[x_neigh, y_neigh, z_neigh,:,-1,-1], size)
                elif y_neigh == y:
                    if   z_neigh > z and x_neigh > x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,:,-1], m_patches[x_neigh, y_neigh, z_neigh,0,:,0], size)
                    elif z_neigh > z and x_neigh < x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,:,-1], m_patches[x_neigh, y_neigh, z_neigh,-1,:,0], size)
                    elif z_neigh < z and x_neigh > x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,:,0], m_patches[x_neigh, y_neigh, z_neigh,0,:,-1], size)
                    elif z_neigh < z and x_neigh < x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,:,0], m_patches[x_neigh, y_neigh, z_neigh,-1,:,-1], size)
                elif z_neigh == z:
                    if   x_neigh > x and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,-1,:], m_patches[x_neigh, y_neigh, z_neigh,0,0,:], size)
                    elif x_neigh > x and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,0,:], m_patches[x_neigh, y_neigh, z_neigh,0,-1,:], size)
                    elif x_neigh < x and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,-1,:], m_patches[x_neigh, y_neigh, z_neigh,-1,0,:], size)
                    elif x_neigh < x and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,0,:], m_patches[x_neigh, y_neigh, z_neigh,-1,-1,:], size)

def find_adjacent_nodes26(adj_mtx, m_patches, x, y, z):
    size =  list(m_patches.shape)
    for z_neigh in range(max(0, z-1), 1+min(m_patches.shape[2]-1, z+1)):
        for y_neigh in range(max(0, y-1), 1+min(m_patches.shape[1]-1, y+1)):
            for x_neigh in range(max(0, x-1), 1+min(m_patches.shape[0]-1, x+1)):
                if z_neigh == z and y_neigh == y and x_neigh == x: 
                    continue
                # check faces
                if x_neigh == x and y_neigh == y:
                    if z_neigh > z:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,:,-1], m_patches[x_neigh, y_neigh, z_neigh, :, :, 0], size)
                    elif z_neigh < z:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,:,0], m_patches[x_neigh, y_neigh, z_neigh,:, :, -1], size)
                elif x_neigh == x and z_neigh == z:
                    if y_neigh > y:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,-1,:], m_patches[x_neigh, y_neigh, z_neigh, :,0,:], size)
                    elif y_neigh < y:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,0,:], m_patches[x_neigh, y_neigh, z_neigh,:,-1,:], size)
                elif y_neigh == y and z_neigh == z:
                    if x_neigh > x:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,:,:], m_patches[x_neigh, y_neigh, z_neigh,0,:,:], size)
                    else:
                        check_face(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,:,:], m_patches[x_neigh, y_neigh, z_neigh,-1,:,:], size)
                #check edges
                elif x_neigh == x:
                    if   z_neigh > z and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,-1,-1], m_patches[x_neigh, y_neigh, z_neigh,:,0,0], size)
                    elif z_neigh > z and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,0,-1], m_patches[x_neigh, y_neigh, z_neigh,:,-1,0], size)
                    elif z_neigh < z and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,-1,0], m_patches[x_neigh, y_neigh, z_neigh,:,0,-1], size)
                    elif z_neigh < z and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,:,0,0], m_patches[x_neigh, y_neigh, z_neigh,:,-1,-1], size)
                elif y_neigh == y:
                    if   z_neigh > z and x_neigh > x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,:,-1], m_patches[x_neigh, y_neigh, z_neigh,0,:,0], size)
                    elif z_neigh > z and x_neigh < x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,:,-1], m_patches[x_neigh, y_neigh, z_neigh,-1,:,0], size)
                    elif z_neigh < z and x_neigh > x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,:,0], m_patches[x_neigh, y_neigh, z_neigh,0,:,-1], size)
                    elif z_neigh < z and x_neigh < x:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,:,0], m_patches[x_neigh, y_neigh, z_neigh,-1,:,-1], size)
                elif z_neigh == z:
                    if   x_neigh > x and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,-1,:], m_patches[x_neigh, y_neigh, z_neigh,0,0,:], size)
                    elif x_neigh > x and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,0,:], m_patches[x_neigh, y_neigh, z_neigh,0,-1,:], size)
                    elif x_neigh < x and y_neigh > y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,-1,:], m_patches[x_neigh, y_neigh, z_neigh,-1,0,:], size)
                    elif x_neigh < x and y_neigh < y:
                        check_edge(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,0,:], m_patches[x_neigh, y_neigh, z_neigh,-1,-1,:], size)
                # check corners
                else:
                    if z_neigh > z and y_neigh > y and x_neigh > x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,-1,-1], m_patches[x_neigh, y_neigh, z_neigh,0,0,0], size)
                    elif z_neigh > z and y_neigh > y and x_neigh < x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,-1,-1], m_patches[x_neigh, y_neigh, z_neigh,-1,0,0], size)
                    elif z_neigh > z and y_neigh < y and x_neigh > x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,0,-1], m_patches[x_neigh, y_neigh, z_neigh,0,-1,0], size)
                    elif z_neigh > z and y_neigh < y and x_neigh < x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,0,-1], m_patches[x_neigh, y_neigh, z_neigh,-1,-1,0], size)
                    
                    elif z_neigh < z and y_neigh > y and x_neigh > x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,-1,0], m_patches[x_neigh, y_neigh, z_neigh,0,0,-1], size)
                    elif z_neigh < z and y_neigh > y and x_neigh < x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,-1,0], m_patches[x_neigh, y_neigh, z_neigh,-1,0,-1], size)
                    elif z_neigh < z and y_neigh < y and x_neigh > x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,-1,0,0], m_patches[x_neigh, y_neigh, z_neigh,0,-1,-1], size)
                    elif z_neigh < z and y_neigh < y and x_neigh < x:
                        check_corner(adj_mtx, x, y, z, x_neigh, y_neigh, z_neigh, m_patches[x,y,z,0,0,0], m_patches[x_neigh, y_neigh, z_neigh,-1,-1,-1], size)
                    else:
                        print("BUG: no category avail.... This shouldnt have happened")
                        raise RuntimeError

def find_adjacency_matrix6(adj_mtx, m_patches, patch_size):
    # naive implementation for connectivity 6...
    # we traverse first in x-axis then on y-axis and then on z-axis.
    # In accordance to node feature creation
    for z in range(0, m_patches.shape[2]):
        for y in range(0, m_patches.shape[1]):
            for x in range(0, m_patches.shape[0]):
                find_adjacent_nodes6(adj_mtx, m_patches, x, y, z)

def find_adjacency_matrix18(adj_mtx, m_patches, patch_size):
    # naive implementation for connectivity 18...
    # we traverse first in x-axis then on y-axis and then on z-axis.
    # In accordance to node feature creation
    for z in range(0, m_patches.shape[2]):
        for y in range(0, m_patches.shape[1]):
            for x in range(0, m_patches.shape[0]):
                find_adjacent_nodes18(adj_mtx, m_patches, x, y, z)

def find_adjacency_matrix26(adj_mtx, m_patches, patch_size):
    # naive implementation for connectivity 26...
    # we traverse first in x-axis then on y-axis and then on z-axis.
    # In accordance to node feature creation
    for z in range(0, m_patches.shape[2]):
        for y in range(0, m_patches.shape[1]):
            for x in range(0, m_patches.shape[0]):
                find_adjacent_nodes26(adj_mtx, m_patches, x, y, z)

def get_adjacency_matrix(volume, patch_size, connectivity):
    
    patches  = get_patches(volume, patch_size)
    n_nodes  = patches.shape[0] * patches.shape[1] * patches.shape[2]
    # later transform it to sparse format
    adj_mtx  = torch.zeros((n_nodes, n_nodes), dtype=torch.int8)

    if connectivity == 6:
        find_adjacency_matrix6(adj_mtx, patches, patch_size)
    elif connectivity == 18: 
        find_adjacency_matrix18(adj_mtx, patches, patch_size)
    elif connectivity == 26:
        find_adjacency_matrix26(adj_mtx, patches, patch_size)
    else:
        print(f'Inside Dataset class: Connectivity {connectivity} incorrect... Exiting')
        raise ValueError
    del patches
    return adj_mtx


#---------- General helper functions
def load_data(path_data, path_splits, fold_id, train_data_name, valid_data_name, test_data_name):
    
    if os.path.exists(path_data) == False:
        print(f'Path to data {path_data} doesnt exists... Exiting')
        raise FileNotFoundError
    names_list = os.listdir(path_data)

    if os.path.exists(path_splits) == False:
        print(f'Path to splits {path_splits} doesnt exists... Exiting')
        raise FileNotFoundError
    names_df   = pd.read_json(path_splits)[fold_id]

    # read training, validation, testing
    # every file should have _tof init image, _seg ground truth, and _mask the coarse mask
    names_dict    = dict()
    training_list = list()
    for tname in names_df[train_data_name]:
        name     = tname.lower()
        tof_list = [i for i in names_list if i.find(name) != -1 and i.find('_tof') != -1 and i.endswith('.h5')]
        msk_list = [i for i in names_list if i.find(name) != -1 and i.find('_mask') != -1 and i.endswith('.h5')]
        seg_list = [i for i in names_list if i.find(name) != -1 and i.find('_seg') != -1 and i.endswith('.h5') and i.find('_seg_real') == -1]
        if tof_list == [] or seg_list ==[] or msk_list == []:
            print(f'Warning: {name} has {tof_list}, {seg_list}, {msk_list}')
            continue
        if len(tof_list)>1 or len(seg_list) > 1 or len(msk_list)>1:
            print(f'Warning: {name} has thr len:{len(tof_list)} and seg len:{len(seg_list)} and msk len:{len(msk_list)}')
        data_pair = {'name': name,
                     'imag': os.path.join(path_data, tof_list[0]),
                     'mask': os.path.join(path_data, msk_list[0]),
                     'segm': os.path.join(path_data, seg_list[0])}
        training_list.append(data_pair)
    names_dict['train'] = training_list

    validation_list = list()
    for tname in names_df[valid_data_name]:
        name     = tname.lower()
        tof_list = [i for i in names_list if i.find(name) != -1 and i.find('_tof') != -1 and i.endswith('.h5')]
        msk_list = [i for i in names_list if i.find(name) != -1 and i.find('_mask') != -1 and i.endswith('.h5')]
        seg_list = [i for i in names_list if i.find(name) != -1 and i.find('_seg') != -1 and i.endswith('.h5') and i.find('_seg_rea') == -1]
        if tof_list == [] or seg_list ==[] or msk_list == []:
            print(f'Warning: {name} has {tof_list}, {seg_list}, {msk_list}')
            continue
        if len(tof_list)>1 or len(seg_list) > 1 or len(msk_list)>1:
            print(f'Warning: {name} has thr len:{len(tof_list)} and seg len:{len(seg_list)} and msk len:{len(msk_list)}')
        data_pair = {'name': name,
                     'imag': os.path.join(path_data, tof_list[0]),
                     'mask': os.path.join(path_data, msk_list[0]),
                     'segm': os.path.join(path_data, seg_list[0])}
        validation_list.append(data_pair)
    names_dict['valid'] = validation_list

    testing_list = list()
    for tname in names_df[test_data_name]:
        name     = tname.lower()
        tof_list = [i for i in names_list if i.find(name) != -1 and i.find('_tof') != -1 and i.endswith('.h5')]
        msk_list = [i for i in names_list if i.find(name) != -1 and i.find('_mask') != -1 and i.endswith('.h5')]
        seg_list = [i for i in names_list if i.find(name) != -1 and i.find('_seg') != -1 and i.endswith('.h5') and i.find('_seg_rea') == -1]
        if tof_list == [] or seg_list ==[] or msk_list == []:
            print(f'Warning: {name} has {tof_list}, {seg_list}, {msk_list}')
            continue
        if len(tof_list)>1 or len(seg_list) > 1 or len(msk_list)>1:
            print(f'Warning: {name} has thr len:{len(tof_list)} and seg len:{len(seg_list)} and msk len:{len(msk_list)}')
        data_pair = {'name': name,
                     'imag': os.path.join(path_data, tof_list[0]),
                     'mask': os.path.join(path_data, msk_list[0]),
                     'segm': os.path.join(path_data, seg_list[0])}
        testing_list.append(data_pair)
    names_dict['test'] = testing_list

    return names_dict
