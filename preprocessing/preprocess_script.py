'''
This is the main step of preprocessing. We assume that the dataset is already cleaned
so all tof files have been in nii format and the segmentation masks are corrected.

The aims of this script are:
1) N4-bias correction
2) Skull stripping
3) Thresholding (either global thresholding or local adaptive thresholding)
4) Reshape & rescale tof images and segmentation masks
5) Save both segmentation and thresholded images in hdf5, nii.gz, .npy format 
in a new folder containing only the images.
'''

import os
import numpy as np
import nibabel as nib
import multiprocessing
import preprocessing_utils as pu

#---------- paths & hyperparameters
# hardcode them for now. Later maybe replace them.
multi_proc                = True
n_threads                 = 4
apply_n4_bias             = False
apply_enchancement        = False
apply_skull_stripping     = False
apply_thresholding_lc     = False
apply_thresholding_gl     = True
global_threshold          = 99.5
apply_rescaling           = False
voxel_size                = np.array([0.3, 0.3, 0.6])
dimensions                = np.array([560, 640, 175]) # or None to leave untouch the initial dimensions
save_logs                 = True
path_to_logs              = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results'
path_to_dataset           = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/skull_stripped'
path_to_save_process_data = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets'
#----------
# paths for testing, not important. They must be deleted in the future
path_to_test              = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/n4_corrected'
#----------

def preprocess_file(file_idx, 
                    subjects_dict, 
                    apply_n4,
                    apply_ench, 
                    apply_sk_strip,
                    global_th_flag,
                    gl_threshold,
                    local_th_flag,
                    save_logs, 
                    path_to_process_data,
                    path_to_logs,
                    apply_resc,
                    new_voxel_size,
                    new_dimensions = None,
                    lock           = None,
                    multipreproc   = False):
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
        tof_img_data = pu.n4_bias_correction(tof_path, name_key, save_logs, path_to_logs, lock, multipreproc)
        # save n4 corrected image and log bias for analysis
        n4_dir       = os.path.join(path_to_process_data, 'n4_corrected')
        pu.create_dir(n4_dir, lock, multipreproc)
        # save image with the segm masks
        pu.save_nii_img(nib.Nifti1Image(tof_img_data, tof_img_aff, tof_img_header), 
                        os.path.join(n4_dir, name_key + '_tof.nii.gz'))
        pu.save_nii_img(nib.Nifti1Image(seg_img_data, seg_img_aff, seg_img_header), 
                        os.path.join(n4_dir, name_key + '_seg.nii.gz'))
        if subjects_dict[name_key]['real_seg'] != []: 
            pu.save_nii_img(nib.Nifti1Image(seg_real_data, seg_real_aff, seg_real_header), 
                            os.path.join(n4_dir, name_key + '_seg_real.nii.gz'))
    if apply_ench:
        ench_dir     = os.path.join(path_to_process_data, 'enchanced')
        pu.create_dir(ench_dir, lock, multipreproc)
        tof_img_data, tof_path = pu.enchance(tof_img_data, tof_img_aff, tof_img_header, name_key, ench_dir)

    if apply_sk_strip:
        sk_dir = os.path.join(path_to_process_data, 'skull_stripped')
        pu.create_dir(sk_dir, lock, multipreproc)
        pu.skull_stripping(tof_path, name_key, sk_dir)
        pu.save_nii_img(nib.Nifti1Image(seg_img_data, seg_img_aff, seg_img_header), 
                        os.path.join(sk_dir, name_key + '_seg.nii.gz'))
        if subjects_dict[name_key]['real_seg'] != []: 
            pu.save_nii_img(nib.Nifti1Image(seg_real_data, seg_real_aff, seg_real_header), 
                            os.path.join(sk_dir, name_key + '_seg_real.nii.gz'))
    
    # apply thresholding
    save_dir = ''
    if global_th_flag:   
        # global threshold v1
        tof_img_data = pu.global_thresholding(tof_img_data, 
                                              np.percentile(tof_img_data, gl_threshold))
        # global threshold v2
        #tof_img_data = pu.global_thresholding2(tof_img_data, gl_threshold)

        save_dir    = os.path.join(path_to_process_data, 
                                   f'pro_data_n4_{apply_n4}_sk_{apply_sk_strip}_gl{gl_threshold}')
    elif local_th_flag:  
        pu.local_thresholding()
        save_dir    = os.path.join(path_to_process_data, 
                                   f'pro_data_n4_{apply_n4}_sk_{apply_sk_strip}_lc')
    else: print(f'Warning: No thresholding applied')

    # reshaping and rescaling
    if apply_resc:
        tof_img_data, tof_img_aff, tof_img_header = pu.adjust_shapes(nib.Nifti1Image(tof_img_data, 
                                                                                    tof_img_aff, 
                                                                                    tof_img_header),
                                                                    new_voxel_size,
                                                                    new_dimensions
                                                                    )
    # save final images ready for training
    if save_dir != '':
        pu.create_dir(save_dir, lock, multipreproc)

        # save images and masks in both hdf5 and nii format
        pu.save_nii_img(nib.Nifti1Image(tof_img_data, tof_img_aff, tof_img_header), 
                        os.path.join(save_dir, name_key + '_tof.nii.gz'))
        #pu.save_hdf5_img(tof_img_data, tof_img_aff, tof_img_header,
        #                 os.path.join(save_dir, name_key + '_tof.h5'))
        pu.save_nii_img(nib.Nifti1Image(seg_img_data, seg_img_aff, seg_img_header), 
                        os.path.join(save_dir, name_key + '_seg.nii.gz'))
        #pu.save_hdf5_img(seg_img_data, seg_img_aff, seg_img_header,
        #                 os.path.join(save_dir, name_key + '_seg.h5'))
        if subjects_dict[name_key]['real_seg'] != []: 
            pu.save_nii_img(nib.Nifti1Image(seg_real_data, seg_real_aff, seg_real_header), 
                            os.path.join(save_dir, name_key + '_seg_real.nii.gz'))
            #pu.save_hdf5_img(seg_real_data, seg_real_aff, seg_img_header,
            #                 os.path.join(save_dir, name_key + '_seg_real.h5'))

def run_process(subjects_tofs_segm_dict, apply_n4_bias, apply_skull_stripping,apply_thresholding_gl,
                                                                          global_threshold,
                                                                          apply_thresholding_lc,
                                                                          save_logs,
                                                                          path_to_save_process_data,
                                                                          path_to_logs,
                                                                          apply_resc,
                                                                          voxel_size_new,
                                                                          dimensions_new,
                                                                          lock,
                                                                          multi_proc,
                                                                          every_n = 4,
                                                                          start_i = 0):
    for idx in range(start_i, len(subjects_tofs_segm_dict), every_n):
        preprocess_file(idx, 
                        subjects_tofs_segm_dict, 
                        apply_n4_bias,
                        apply_enchancement,
                        apply_skull_stripping,
                        apply_thresholding_gl,
                        global_threshold,
                        apply_thresholding_lc,
                        save_logs,
                        path_to_save_process_data,
                        path_to_logs,
                        apply_resc,
                        voxel_size_new,
                        dimensions_new,
                        lock,
                        multi_proc)

if __name__ == '__main__':
    '''
    # this part of code is only used for devs. It has to be removed in the future
    subjects_tofs_segm_dict_old = pu.locate_tof_mask_paths(path_to_dataset)
    old_keys                    = list(subjects_tofs_segm_dict_old.keys())
    subjects_tofs_segm_dict2    = pu.locate_tof_mask_paths(path_to_test)
    new_keys                    = list(subjects_tofs_segm_dict2.keys())
    subjects_tofs_segm_dict     = dict()
    new_keys = [key for key in old_keys if key not in new_keys]
    for key in new_keys:    subjects_tofs_segm_dict[key] = subjects_tofs_segm_dict_old[key]
    '''
    subjects_tofs_segm_dict = pu.locate_tof_mask_paths(path_to_dataset)
    if multi_proc == False:
        for idx in range(len(subjects_tofs_segm_dict)):
            preprocess_file(idx, 
                            subjects_tofs_segm_dict, 
                            apply_n4_bias,
                            apply_enchancement,
                            apply_skull_stripping,
                            apply_thresholding_gl,
                            global_threshold,
                            apply_thresholding_lc,
                            save_logs,
                            path_to_save_process_data,
                            path_to_logs,
                            apply_rescaling,
                            voxel_size,
                            dimensions,
                            )
    else:
        # use a Manager object to create a shared 'Lock' object
        manager    = multiprocessing.Manager()
        lock       = manager.Lock()
        split_dif  = n_threads
        start_from = 0
        split_id   = 0
        ps         = []
        for k in range(start_from + split_id*split_dif, start_from + split_dif*(split_id+1)):
            ps.append(multiprocessing.Process(target=run_process, args = (subjects_tofs_segm_dict, 
                                                                          apply_n4_bias, 
                                                                          apply_skull_stripping,
                                                                          apply_thresholding_gl,
                                                                          global_threshold,
                                                                          apply_thresholding_lc,
                                                                          save_logs,
                                                                          path_to_save_process_data,
                                                                          path_to_logs,
                                                                          apply_rescaling,
                                                                          voxel_size,
                                                                          dimensions,
                                                                          lock,
                                                                          multi_proc,
                                                                          n_threads,
                                                                          k
                                                                          )))
        for k in range(len(ps)):
            ps[k].start()
        for k in range(len(ps)):
            ps[k].join()
    
    print("END OF PREPROC")
