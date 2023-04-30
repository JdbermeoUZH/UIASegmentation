import os
import pandas as pd

def load_split_dict(path_data, path_splits, fold_id, train_data_name, valid_data_name, test_data_name):
    
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
