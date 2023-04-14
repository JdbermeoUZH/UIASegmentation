'''
As the name suggest this script update the initial dataset. The script applies 
the following:
    1) Save tof.nrrd or tof.dicom image in nii.gz
    2) Label correction in segmentation masks
    3) Create two semgentation masks one with the labels and the second
    with the true values only for the vessels, the rest are set to 0
'''

import os
import shutil
import pandas as pd
import preprocessing_utils as pu

path_to_initial_dataset = '/scratch/kvergopoulos/SemesterProject/datasets/USZ_BrainArtery_Originals'
path_to_updated_dataset = '/scratch/kvergopoulos/SemesterProject/datasets/USZ_BrainArtery_updated'


def process_tofs_func(paths_list,temp_folder_path):

    for idx, mri_path in enumerate(paths_list):
        print(f'processing: {idx}, {mri_path}')
        assert os.path.isdir(mri_path)
        tof_nrrd_list = list()
        tof_nii_list  = list()
        dicomdir_ext  = False
        for t_file in os.listdir(mri_path):
            if t_file == 'DICOMDIR':
                dicomdir_ext = True
                continue
            if t_file.endswith(".nrrd") and t_file.lower().find("segmentation") == -1:
                tof_nrrd_list.append(t_file)
            elif (t_file.endswith(".nifti") or t_file.endswith(".nii") or \
                 t_file.endswith(".nii.gz")) and t_file.lower().find("tof") != -1:
                tof_nii_list.append(t_file)
        if tof_nii_list != []:
            print(f'tof file already in nii {tof_nii_list}')
        elif tof_nrrd_list != []:
            # convert nrrd to nifti
            if len(tof_nrrd_list) == 1:
                pu.conv_nrrd2nifti(os.path.join(mri_path, tof_nrrd_list[0]))
            else:
                print(f'Multiple tof files {tof_nrrd_list} choose the first one')
                pu.conv_nrrd2nifti(os.path.join(mri_path, tof_nrrd_list[0]))
        elif dicomdir_ext == True:
            # convert dicomdir to nifti
            pu.conv_dicom2nifti(mri_path, temp_folder_path)
        else:
            raise Exception(f"No option for tof file found in {mri_path}")

def process_segm_func(paths_list, path_to_original, temp_folder_path):

    # load groundtruth class table 
    class_table = pd.read_excel(os.path.join(path_to_original, 'class_labels.xlsx'))

    for idx, mri_path in enumerate(paths_list):
        print(f'processing: {idx}, {mri_path}')
        assert os.path.isdir(mri_path)

        segm_masks_list = [i for i in os.listdir(mri_path) if i.lower().find('segmentation')!=-1 \
                           and i.lower().find('-label') == -1 and i.lower().endswith('.nrrd')]
        assert len(segm_masks_list) == 1

        ### the following code was adopted from previous works
        file_classes       = pu.read_classes_from_nrrd(os.path.join(mri_path, segm_masks_list[0]))
        mapping            = pu.match_labels(file_classes, class_table)
        mapping_pd         = pd.DataFrame.from_dict(mapping, orient='index')
        mapping_pd.columns = ['class_id', 'file_id', 'class_name', 'name_in_file']
        mapping_pd.to_csv(os.path.join(mri_path, 'label_assignment2.csv'),index=False)
        
        # compute and save the corrected segmentation mask
        segm_masks_list2 = [i for i in os.listdir(mri_path) if i.lower().find('segmentation')!=-1 \
                           and (i.lower().endswith('.nii.gz') or i.lower().endswith('.nii'))\
                           and i.lower().find('corrected_') == -1]
        assert len(segm_masks_list2) == 1

        pu.create_corrected_mask(mri_path,
                                 os.path.join(mri_path, 'label_assignment2.csv'),
                                 os.path.join(mri_path, segm_masks_list2[0]))
        ###

def update_fix_dataset(path_to_dataset, path_to_original, process_tofs = True, process_segm = True):
    
    # create a temporary folder which will be deleted at the end
    temp_folder_path = os.path.join(path_to_dataset, 'temp_folder')
    if os.path.exists(temp_folder_path) == False: os.makedirs(temp_folder_path)

    paths_list = pu.get_subjects_folders(path_to_dataset)
    assert len(paths_list) == 62
    
    # process tof files
    if process_tofs == True:    process_tofs_func(paths_list, temp_folder_path)
    # process segmentation masks
    if process_segm == True:    process_segm_func(paths_list, path_to_original, temp_folder_path)

    # delete temporary folder
    shutil.rmtree(temp_folder_path)

# run the script
update_fix_dataset(path_to_updated_dataset, path_to_initial_dataset, process_tofs = True, process_segm = True)