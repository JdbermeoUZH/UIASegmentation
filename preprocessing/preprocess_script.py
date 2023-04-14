'''
This is the main step of preprocessing. We assume that the dataset is already cleaned
so all tof files have been in nii format and the segmentation masks are corrected.

The aims of this script are:
1) N4-bias correction
2) Skull stripping
3) Thresholding (either global thresholding or local adaptive thresholding)
4) Validate the results of thresholding
5) Reshape & rescale tof images and segmentation masks
6) Save both segmentation and thresholded images in hdf5, nii.gz, .npy format 
in a new folder containing only the images.
'''

import os
import numpy as np
import nibabel as nib
import multiprocessing
import preprocessing_utils as pu


#---------- paths & hyperparameters
# hardcode them for now. Later maybe replace them.
n_threads                 = 1
apply_n4                  = True
apply_skull_stripping     = True
apply_thresholding_gl     = True
apply_thresholding_lc     = False
save_figs_plots           = True
file_formats              = 'all' #dont know yet which format is usuful so save each image in .nii, hdf5, npy
path_to_figs              = '/home/kostas/Desktop/scratch/SemesterProject/intermediate_results'
path_to_dataset           = '/scratch/kvergopoulos/SemesterProject/datasets/USZ_BrainArtery_updated'
path_to_save_process_data = '/home/kostas/Desktop/scratch/SemesterProject/datasets'
#----------

def preprocess_file(file_idx, 
                    subjects_dict, 
                    apply_n4, 
                    save_figs_plots, 
                    path_to_save_process_data, 
                    path_to_figs):
    ###
    name_key = list(subjects_dict.keys())[file_idx]
    tof_path = subjects_dict[name_key]['tof'][0]
    seg_path = subjects_dict[name_key]['seg'][0]
    if len(subjects_dict[name_key]['real_seg']) >0: real_seg_path = subjects_dict[name_key]['real_seg'][0]
    else: real_seg_path = ''
    print(f'Processing file with index {file_idx} and name {name_key}')

    # load initial image, segmentation mask and real segmentation mask
    tof_img_data, tof_img_aff, tof_img_header = pu.read_all_from_nii(tof_path)
    seg_img_data, seg_img_aff, seg_img_header = pu.read_all_from_nii(seg_path) 
    if subjects_dict[name_key]['real_seg'] != []:   
        seg_real_data, seg_real_aff, seg_real_header = pu.read_all_from_nii(real_seg_path)
    
    if apply_n4:
        tof_img_data = pu.n4_bias_correction(tof_path, save_figs_plots, path_to_figs)


subjects_tofs_segm_dict = pu.locate_tof_mask_paths(path_to_dataset)
if n_threads == 1:
    for idx in range(len(subjects_tofs_segm_dict)):
        preprocess_file(idx, 
                        subjects_tofs_segm_dict, 
                        apply_n4,
                        save_figs_plots,
                        path_to_save_process_data,
                        path_to_figs)
        break
else:
    # later
    pass