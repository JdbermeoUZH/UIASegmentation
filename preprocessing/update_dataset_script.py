'''
As the name suggest this script update the initial dataset. The script applies 
the following:
    1) Save tof.nrrd or tof.dicom image in nii.gz
    2) Label correction in segmentation masks
    3) Create two semgentation masks one with the labels and the second
    with the true values only for the vessels. The rest is 0
'''

import os
import preprocessing_utils as pu

path_to_updated_dataset = '/scratch/kvergopoulos/SemesterProject/datasets/USZ_BrainArtery_updated'

def update_fix_dataset(path_to_dataset):
    
    paths_list = pu.get_subjects_folders(path_to_dataset)

    # process tof files
    for mri_path in paths_list:
        print(f'processing {mri_path}')
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
            pu.conv_dicom2nifti(mri_path)
        else:
            raise Exception(f"No option for tof file found in {mri_path}")

    # process segmentation masks
    # find the segmentation mask and correct it
    
update_fix_dataset(path_to_updated_dataset)