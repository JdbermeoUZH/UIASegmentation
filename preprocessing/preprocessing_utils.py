'''

This script contains helper functions that are used in all stages of the preprocessing

'''

import os
import nrrd
import shutil
import dicom2nifti
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from difflib import SequenceMatcher


def get_subjects_folders(path_to_data):
    '''
    This function iterate through all the subfolders inside the path_to_data folder
    and returns a list with all the paths.

    Parameters
    ----------
    path_to_data: The path for the dataset

    Returns
    -------
    list of paths to each subject
    '''
    subjects_paths_list = list()
    for mri_file in os.listdir(path_to_data):
        mri_file_path = os.path.join(path_to_data, mri_file)
        if os.path.isdir(mri_file_path) and not mri_file.startswith('.') \
            and not mri_file.find("unlabelled") != -1 and not mri_file.find("temp_folder") != -1:
            subjects_paths_list.append(mri_file_path)
    return subjects_paths_list

def find_types_of_aneurysm(aneurysm_classes, name_components):
    '''
    Parameters
    ----------
    aneurysm_classes: all classes of aneurysm
    name_components: name of file splitted in tokens

    Returns
    -------
    list of all types of aneurysm
    '''
    types_of_aneur = list()
    for aneur in aneurysm_classes:
        for comp in name_components:
            if aneur == comp or comp.startswith(aneur) or comp.endswith(aneur):
                if aneur == 'ica' and (comp.startswith('pica') or comp.endswith('pica')): continue
                types_of_aneur.append(aneur)
                break
    return types_of_aneur

#### functions for correcting labels. Adopted from previous work
def read_classes_from_nrrd(path):
    data, header = nrrd.read(path)
    segment_keys = {}
    for key in header.keys():
        if key.startswith('Segment') and key[7].isdigit():
            segid, cont = key.split('_')
            index = int(segid[7:])
            if index not in segment_keys.keys():    segment_keys[index] = {}
            segment_keys[index][cont] = header[key]
    label_list = pd.DataFrame.from_dict(segment_keys, orient='index')
    label_list = label_list[['Name', 'LabelValue']].astype({'LabelValue':int}).sort_values(by=['LabelValue'])
    return label_list.rename(columns={'Name':'name', 'LabelValue':'label id'})

def match_labels(color_table, class_table):
    mapping        = {}
    i              = 0
    js             = list(range(class_table.shape[0]))
    matches_ratio  = []

    while len(js) != 0 and i < color_table.shape[0]:
        matches    = [SequenceMatcher(None, class_table['name'].iloc[j], color_table['name'].iloc[i]).ratio() for j in js]
        j          = js[matches.index(max(matches))]
        mapping[j] = [class_table['label id'].iloc[j], color_table['label id'].iloc[i], class_table['name'].iloc[j], color_table['name'].iloc[i]] 
        i          = i+1
        js.remove(j)
        matches_ratio.append(max(matches))
    
    if np.mean(matches_ratio) < 0.75:
        print('Warning: Mapping did not work correctly', np.mean(matches_ratio))
    
    for j in js:
        mapping[j] = [class_table['label id'].iloc[j], None, class_table['name'].iloc[j], None]
    return mapping

def correct_labels(data, mapping):
    if isinstance(mapping, pd.DataFrame):
        label_mapping = mapping[['file_id', 'class_id']].dropna(axis='index', how='any').set_index('file_id').to_dict()['class_id']
    else:
        label_mapping = mapping 
    return map_labels(data, label_mapping).astype(int)

def map_labels(data, label_mapping):
    l_keys = []
    for key in label_mapping.keys():
        if label_mapping[key] != key:
            l_keys.append(key)

    data_new = np.copy(data)
    x_dim, y_dim, z_dim = data_new.shape
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if int(data_new[x,y,z]) in l_keys:
                    data_new[x,y,z] = int(label_mapping[int(data_new[x,y,z])])
    return data_new

def create_corrected_mask(mri_path, mapping_path, wrong_mask_path):
    print('Reading label mapping from', mapping_path)
    label_mapping = pd.read_csv(mapping_path, dtype={'class_id':int,
                                                     'file_id':float,
                                                     'class_name':str,
                                                     'name_in_file':str})
    print('Reading wrong mask from', wrong_mask_path)
    nii_wrong_mask = nib.load(wrong_mask_path)
    
    data_corr      = correct_labels(nii_wrong_mask.get_fdata(), label_mapping) 

    affine_new     = nii_wrong_mask.affine.copy()
    # sanity check. Affine matrix from wrong mask and init image match
    tof_list       = [i for i in os.listdir(mri_path) if i.lower().find('tof')!=-1 \
                      and i.lower().endswith('.nii.gz')]
    assert len(tof_list) == 1
    tof_image = nib.load(os.path.join(mri_path, tof_list[0]))
    print(affine_new)
    print(tof_image.affine)
    print(tof_image.affine == affine_new)

    #nii_corr_mask  = nib.Nifti1Image(data_corr.astype(int), 
    #                                 affine_new, 
    #                                 nii_wrong_mask.header, 
    #                                 dtype = nii_wrong_mask.get_data_dtype())
    
    #head, tail = os.path.split(wrong_mask_path)
    #nib.save(nii_corr_mask, os.path.join(head, 'corrected_' + tail))

    
###

def conv_nrrd2nifti(path):
    head, _         = os.path.split(path)
    path_components = path.split('/')
    tof_path        = os.path.join(head, path_components[-2].lower() + '_tof.nii.gz')
    
    tof_img         = sitk.ReadImage(path)
    sitk.WriteImage(tof_img, tof_path)
    print(f'Saving to: {tof_path}')

def conv_dicom2nifti(path, temp_folder = None):
    if temp_folder != None:
        path_components = path.split('/')
        save_path       = os.path.join(temp_folder, path_components[-1] + '_tof_folder')
        print("TEST SAVE PATH", save_path)
        if os.path.exists(save_path) == False:  os.makedirs(save_path)
        dicom2nifti.convert_directory(path, save_path)
        files = list(os.listdir(save_path))
        assert len(files) == 1
        tof_path = os.path.join(path, path_components[-1].lower() + '_tof.nii.gz')
        shutil.copy(os.path.join(save_path, files[0]), tof_path)
        shutil.rmtree(save_path)
        print(f'Saving from dicom to nifti: {tof_path}')