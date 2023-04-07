'''

This script contains helper functions that are used in all stages of the preprocessing

'''

import os


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
            and not mri_file.find("unlabelled") != -1:
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

def conv_nrrd2nifti(path_test):
    pass

def conv_dicom2nifti(path_test):
    pass