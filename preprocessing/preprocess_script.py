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
apply_n4_bias             = True
apply_skull_stripping     = False
apply_thresholding_gl     = False
global_threshold          = 98.5
apply_thresholding_lc     = False
save_logs                 = True
file_formats              = '.nii.gz' #dont know yet which format is usuful so save each image in .nii, hdf5, npy
path_to_logs              = '/scratch/kvergopoulos/SemesterProject/intermediate_results'
path_to_dataset           = '/scratch/kvergopoulos/SemesterProject/datasets/USZ_BrainArtery_updated'
path_to_save_process_data = '/scratch/kvergopoulos/SemesterProject/datasets'
#----------

def preprocess_file(file_idx, 
                    subjects_dict, 
                    apply_n4, 
                    apply_sk_strip,
                    global_th_flag,
                    gl_threshold,
                    local_th_flag,
                    save_logs, 
                    path_to_process_data,
                    path_to_logs):
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
        tof_img_data = pu.n4_bias_correction(tof_path, name_key, save_logs, path_to_logs)
        # save n4 corrected image
        n4_dir       = os.path.join(path_to_process_data, 'n4_corrected')
        if os.path.exists(n4_dir) == False: os.makedirs(n4_dir)
        # save image with the segm masks
        pu.save_nii_img(nib.Nifti1Image(tof_img_data, tof_img_aff, tof_img_header), 
                        os.path.join(n4_dir, name_key + '_n4_corr.nii.gz'))
        pu.save_nii_img(nib.Nifti1Image(seg_img_data, seg_img_aff, seg_img_header), 
                        os.path.join(n4_dir, name_key + '_seg.nii.gz'))
        if subjects_dict[name_key]['real_seg'] != []: 
            pu.save_nii_img(nib.Nifti1Image(seg_real_data, seg_real_aff, seg_real_header), 
                            os.path.join(n4_dir, name_key + '_seg_real.nii.gz'))
    
    if apply_sk_strip:
        print("fuck youuuuu")
    
    # apply thresholding
    save_dir = ''
    if global_th_flag:   
        tof_img_data = pu.global_thresholding(tof_img_data, 
                                              np.percentile(tof_img_data, gl_threshold))
        save_dir    = os.path.join(path_to_process_data, 
                                   f'pro_data_n4_{apply_n4}_sk_{apply_sk_strip}_gl{gl_threshold}')
    elif local_th_flag:  
        pu.local_thresholding()
        save_dir    = os.path.join(path_to_process_data, 
                                   f'pro_data_n4_{apply_n4}_sk_{apply_sk_strip}_lc')
    else: print(f'Warning: No thresholding applied')

    # reshaping and rescaling

    # save final images ready for training
    if save_dir != '':
        if os.path.exists(save_dir) == False: os.makedirs(save_dir)
        # save images and masks in both hdf5 and nii format
        pu.save_nii_img(nib.Nifti1Image(tof_img_data, tof_img_aff, tof_img_header), 
                        os.path.join(save_dir, name_key + '_.nii.gz'))
        pu.save_hdf5_img(tof_img_data, tof_img_aff, tof_img_header,
                         os.path.join(save_dir, name_key + '_.h5'))
        pu.save_nii_img(nib.Nifti1Image(seg_img_data, seg_img_aff, seg_img_header), 
                        os.path.join(save_dir, name_key + '_seg.nii.gz'))
        pu.save_hdf5_img(seg_img_data, seg_img_aff, seg_img_header,
                         os.path.join(save_dir, name_key + '_seg.h5'))
        if subjects_dict[name_key]['real_seg'] != []: 
            pu.save_nii_img(nib.Nifti1Image(seg_real_data, seg_real_aff, seg_real_header), 
                            os.path.join(save_dir, name_key + '_seg_real.nii.gz'))
            pu.save_hdf5_img(seg_real_data, seg_real_aff, seg_img_header,
                             os.path.join(save_dir, name_key + '_seg_real.h5'))

subjects_tofs_segm_dict = pu.locate_tof_mask_paths(path_to_dataset)
if n_threads == 1:
    for idx in range(len(subjects_tofs_segm_dict)):
        preprocess_file(idx, 
                        subjects_tofs_segm_dict, 
                        apply_n4_bias,
                        apply_skull_stripping,
                        apply_thresholding_gl,
                        global_threshold,
                        apply_thresholding_lc,
                        save_logs,
                        path_to_save_process_data,
                        path_to_logs)
else:
    # later
    pass