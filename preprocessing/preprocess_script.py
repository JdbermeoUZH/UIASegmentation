'''
This is the main step of preprocessing. We assume that the dataset is already cleaned
so all tof files have been in nii format and the segmentation masks are corrected.

The aims of this script are:
1) Remove 10 lower brain slices
2) N4-bias correction
3) Skull stripping
4) Thresholding (either global thresholding or local adaptive thresholding)
5) Reshape & rescale tof images and segmentation masks
6) Save init image, segmentation and thresholded images in hdf5 format and also some stats
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
n_threads                 = 2
remove_lower_slices       = False
number_of_slices          = 10
apply_n4_bias             = False
apply_enchancement        = False
apply_skull_stripping     = False
skull_stripping_tool      = 'bet' # you can choose between 'bet' and ''
apply_thresholding_lc     = False
apply_thresholding_gl     = False
global_threshold          = 99.5
apply_rescaling           = True
voxel_size                = np.array([0.3, 0.3, 0.6]) # hyper parameters to be set
dimensions                = np.array([560, 640, 160]) # or None to leave untouch the initial dimensions
save_logs                 = True

path_to_logs              = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results'
path_to_dataset           = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results/images_intermediate_folder/global_thresholded_99.5'
path_to_save_process_data = '/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN'
#----------

def preprocess_file(file_idx,
                    subjects_dict,
                    remove_slices,
                    n_slices,
                    apply_n4,
                    apply_ench,
                    apply_sk_strip,
                    sk_strip_tool,
                    apply_lc_flag,
                    apply_gl_flag,
                    gl_threshold,
                    apply_resc,
                    new_voxel_size,
                    new_dimensions,
                    save_logs_flag,
                    path_logs,
                    path_temp_preproc,
                    path_process_data,
                    lock           = None,
                    multipreproc   = False
                    ):
    ###
    name_key      = list(subjects_dict.keys())[file_idx]
    init_img_path = subjects_dict[name_key]['tof'][0]
    gt_seg_path   = subjects_dict[name_key]['seg'][0]
    if len(subjects_dict[name_key]['mask']) >0: 
        proc_mask_path = subjects_dict[name_key]['mask'][0]
    else: proc_mask_path = ''
    print(f'Processing file with index {file_idx} and name {name_key}')

    if remove_slices:
        rem_dir       = os.path.join(path_temp_preproc, 'cropped_images')
        pu.create_dir(rem_dir, lock, multipreproc)
        init_img_path, \
        gt_seg_path,   \
        proc_mask_path = pu.remove_slices(name_key, 
                                          init_img_path, 
                                          gt_seg_path, 
                                          proc_mask_path, 
                                          n_slices, 
                                          rem_dir)
    if apply_n4:
        n4_dir       = os.path.join(path_temp_preproc, 'n4_corrected')
        pu.create_dir(n4_dir, lock, multipreproc)
        init_img_path, \
        gt_seg_path,   \
        proc_mask_path = pu.n4_bias_correction_interface(name_key, 
                                                         init_img_path, 
                                                         gt_seg_path, 
                                                         proc_mask_path, 
                                                         n4_dir,
                                                         save_logs_flag,
                                                         path_logs,
                                                         lock,
                                                         multipreproc)
    if apply_ench:
        ench_dir     = os.path.join(path_temp_preproc, 'enchanced')
        pu.create_dir(ench_dir, lock, multipreproc)
        init_img_path, \
        gt_seg_path,   \
        proc_mask_path = pu.enchance_interface(name_key, 
                                               init_img_path, 
                                               gt_seg_path, 
                                               proc_mask_path, 
                                               ench_dir,
                                               save_logs_flag,
                                               path_logs,
                                               lock,
                                               multipreproc)
    if apply_sk_strip:
        # apply skull stripped not only to mask but to init image
        init_img_skull_stip = False
        sk_dir = os.path.join(path_temp_preproc, 'skull_stripped')
        pu.create_dir(sk_dir, lock, multipreproc)
        if sk_strip_tool == 'bet':
            init_img_path, \
            gt_seg_path,   \
            proc_mask_path = pu.skstrip_bet_interface(name_key, 
                                                      init_img_path, 
                                                      gt_seg_path, 
                                                      proc_mask_path, 
                                                      sk_dir,
                                                      save_logs_flag,
                                                      path_logs,
                                                      lock,
                                                      multipreproc,
                                                      init_img_skull_stip)
        else:
            # implement another skull stip method
            pass
    
    if apply_lc_flag:
        # implement one of the several tested local threshold methods
        pass
    if apply_gl_flag:
        threshold_version = 1
        th_dir = os.path.join(path_temp_preproc, f'global_thresholded_{gl_threshold}')
        pu.create_dir(th_dir, lock, multipreproc)
        init_img_path, \
        gt_seg_path,   \
        proc_mask_path = pu.gl_threshold_interface(name_key, 
                                                   init_img_path, 
                                                   gt_seg_path,
                                                   proc_mask_path, 
                                                   th_dir,
                                                   save_logs_flag,
                                                   path_logs,
                                                   threshold_version,
                                                   gl_threshold,
                                                   lock,
                                                   multipreproc)
    
    final_dir = os.path.join(path_process_data, 'hdf5_dataset')
    pu.create_dir(final_dir, lock, multipreproc)
    pu.save_for_deep_learning(name_key,
                              init_img_path,
                              gt_seg_path,
                              proc_mask_path,
                              final_dir,
                              apply_resc,
                              new_voxel_size,
                              new_dimensions,
                              lock,
                              multipreproc)

def run_process(subjects_tofs_segm_dict,
                remove_lower_slices,
                number_of_slices,
                apply_n4_bias, 
                apply_enchancement,
                apply_skull_stripping,
                skull_stripping_tool,
                apply_thresholding_lc,
                apply_thresholding_gl,
                global_threshold,
                apply_rescaling,
                voxel_size,
                dimensions,
                save_logs,
                path_to_logs,
                path_to_temp_preproc,
                path_to_save_process_data,
                lock,
                multi_proc,
                every_n = 4,
                start_i = 0):
    
    for idx in range(start_i, len(subjects_tofs_segm_dict), every_n):
        preprocess_file(idx,
                        subjects_tofs_segm_dict,
                        remove_lower_slices,
                        number_of_slices,
                        apply_n4_bias,
                        apply_enchancement,
                        apply_skull_stripping,
                        skull_stripping_tool,
                        apply_thresholding_lc,
                        apply_thresholding_gl,
                        global_threshold,
                        apply_rescaling,
                        voxel_size,
                        dimensions,
                        save_logs,
                        path_to_logs,
                        path_to_temp_preproc,
                        path_to_save_process_data,
                        lock,
                        multi_proc)  

if __name__ == '__main__':

    # create temp folder to save all intermediate images
    path_to_temp_preproc = os.path.join(path_to_logs, 'images_intermediate_folder')
    if os.path.exists(path_to_temp_preproc) == False: os.makedirs(path_to_temp_preproc)

    # load the dataset
    # for each subject must be a tof file and a ground truth segmentation
    subjects_tofs_segm_dict = pu.locate_tof_mask_paths(path_to_dataset)
    
    if multi_proc == False:
        for idx in range(len(subjects_tofs_segm_dict)):
            preprocess_file(idx,
                            subjects_tofs_segm_dict,
                            remove_lower_slices,
                            number_of_slices,
                            apply_n4_bias,
                            apply_enchancement,
                            apply_skull_stripping,
                            skull_stripping_tool,
                            apply_thresholding_lc,
                            apply_thresholding_gl,
                            global_threshold,
                            apply_rescaling,
                            voxel_size,
                            dimensions,
                            save_logs,
                            path_to_logs,
                            path_to_temp_preproc,
                            path_to_save_process_data
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
                                                                          remove_lower_slices,
                                                                          number_of_slices,
                                                                          apply_n4_bias, 
                                                                          apply_enchancement,
                                                                          apply_skull_stripping,
                                                                          skull_stripping_tool,
                                                                          apply_thresholding_lc,
                                                                          apply_thresholding_gl,
                                                                          global_threshold,
                                                                          apply_rescaling,
                                                                          voxel_size,
                                                                          dimensions,
                                                                          save_logs,
                                                                          path_to_logs,
                                                                          path_to_temp_preproc,
                                                                          path_to_save_process_data,
                                                                          lock,
                                                                          multi_proc,
                                                                          n_threads,
                                                                          k
                                                                          )))
        
        for k in range(len(ps)):    ps[k].start()
        for k in range(len(ps)):    ps[k].join()

    # delete the temp folder
    #########################
    # exit
    print("END OF PREPROCESSING")
