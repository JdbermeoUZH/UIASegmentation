'''
This script computes several stats.
Precisely it computes:
    1) Voxel sizes
    2) Image shapes & resolutions
'''
import os
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import preprocessing_utils as pu


#---------- paths 
# hardcode them for now. Later maybe replace them.
path_to_inter_results   = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results/figs'
path_to_dataset_updated = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results/images_intermediate_folder/cropped_images'
#---------- 

#---------- def functions
def calculate_distribution_resolution(path_to_data, path_save_results, round_decimal = 0):
    subjects_tofs_segm_dict = pu.locate_tof_mask_paths(path_to_data)
    voxel_sizes  = list()
    tofs_sizes   = list()
    resol_sizes  = list()

    for key in subjects_tofs_segm_dict:
        tof_images = subjects_tofs_segm_dict[key]['tof']
        seg_images = subjects_tofs_segm_dict[key]['seg']
        # look both tof and seg image in order to assert that 
        # both images have the same vox size and same dimensions
        if len(tof_images) > 1:
            # this should not happen, if everything works correctly
            print(f'Warning: There are multiple tof images in {tof_images}, choose the first one')
        tof_image = tof_images[0]
        if len(seg_images)>1:
            # this should not happen, if everything works correctly
            print(f'Warning: There are multiple segmentation images in {seg_images}, choose the first one')  
        seg_image = seg_images[0]

        tof_voxel = pu.read_voxel_from_nii(tof_image)
        tof_size  = pu.read_dims_from_nii(tof_image)
        seg_voxel = pu.read_voxel_from_nii(seg_image)
        seg_size  = pu.read_dims_from_nii(seg_image)

        assert tof_size == seg_size
        assert len(tof_size) == len(seg_size) == len(seg_voxel) == len(tof_voxel) == 3
        try:
            np.testing.assert_almost_equal(tof_voxel, seg_voxel, decimal = 6, err_msg = 'deviation < 6th decimal')
        except AssertionError as msg:
            print(msg)
            print("tof image voxel", tof_voxel)
            print("seg image voxel", seg_voxel)
        
        if round_decimal > 0: tof_voxel = tuple(np.round(tof_voxel, round_decimal))
        resol = tuple([tof_size[0]*tof_voxel[0], tof_size[1]*tof_voxel[1], tof_size[2]*tof_voxel[2]])
        if round_decimal > 0: resol = tuple(np.round(resol, round_decimal))
        voxel_sizes.append(tuple(tof_voxel))
        tofs_sizes.append(tuple(tof_size))
        resol_sizes.append(tuple(resol))
    
    assert len(voxel_sizes)==len(tofs_sizes) == len(resol_sizes) == 62
    # plot resolutions and tofs
    counter_tof_sizes   = Counter(tofs_sizes)
    sizes, sizes_counts = zip(*counter_tof_sizes.items())
    sorted_idxes        = np.argsort(np.prod(sizes, axis=1))
    sizes_sorted        = [sizes[i] for i in sorted_idxes] 
    sizes_counts_sorted = [sizes_counts[i] for i in sorted_idxes]
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x = sizes_counts_sorted, y = [str(i) for i in sizes_sorted], ax=ax1)
    ax1.set_xlabel('Frequencies')
    ax1.set_ylabel('Dimensions order by their volumes')
    ax1.set_title('Histogram of Dimension sizes')
    fig1.savefig(os.path.join(path_save_results, 'dimensions_dist_v2.png'), bbox_inches='tight')

    counter_resolutions   = Counter(resol_sizes)
    resols, resols_counts = zip(*counter_resolutions.items())
    sorted_idxes_resols   = np.argsort(np.prod(resols, axis=1)) 
    resols_sorted         = [resols[i] for i in sorted_idxes_resols]
    resols_counts_sorted  = [resols_counts[i] for i in sorted_idxes_resols]

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(x = resols_counts_sorted, y = [str(i) for i in resols_sorted], ax=ax2)
    ax2.set_xlabel('Frequencies')
    ax2.set_ylabel('Resolutions sorted')
    ax2.set_title('Histogram of Resolutions')
    fig2.savefig(os.path.join(path_save_results, 'resolutions_dist_v2.png'), bbox_inches='tight')



def calculate_distribution_voxelsizes(path_to_data, path_save_results, round_decimal = 0):
    paths_list   = pu.get_subjects_folders(path_to_data)
    voxel_sizes  = list()

    for mri_path in paths_list:
        mri_files  = list(os.listdir(mri_path))
        tof_images = [i for i in mri_files if i.lower().endswith('_tof.nii.gz')]
        seg_images = [i for i in mri_files if i.lower().find('corrected_') != -1 and i.lower().find('real_corrected') == -1]
        # look both tof and seg image in order to assert that both images have the same vox size.
        if len(tof_images) > 1:
            # this should not happen, if everything works correctly
            print(f'Warning: There are multiple tof images in {tof_images}, choose the first one')
        tof_image = tof_images[0]
        if len(seg_images)>1:
            # this should not happen, if everything works correctly
            print(f'Warning: There are multiple segmentation images in {seg_images}, choose the first one')  
        seg_image = seg_images[0]

        tof_voxel = pu.read_voxel_from_nii(os.path.join(mri_path, tof_image))
        seg_voxel = pu.read_voxel_from_nii(os.path.join(mri_path, seg_image))
        try:
            np.testing.assert_almost_equal(tof_voxel, seg_voxel, decimal = 6, err_msg = 'deviation < 6th decimal')
        except AssertionError as msg:
            print(msg)
            print("tof image voxel", tof_voxel)
            print("seg image voxel", seg_voxel)
        if round_decimal > 0: voxel_sizes.append(tuple(np.round(tof_voxel, round_decimal)))
        else: voxel_sizes.append(tof_voxel)
    
    assert len(voxel_sizes)==62
    counter        = Counter(voxel_sizes) 
    voxels, counts = zip(*counter.items())
    sorted_idxes   = np.argsort(np.prod(voxels, axis=1))
    counts_sorted  = [counts[i] for i in sorted_idxes]
    voxels_sorted  = [voxels[i] for i in sorted_idxes]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x = counts_sorted, y = [str(i) for i in voxels_sorted], ax=ax)
    ax.set_xlabel('Frequencies')
    ax.set_ylabel('Voxel sizes order by their volumes')
    ax.set_title('Histogram of Voxel size')
    fig.savefig(os.path.join(path_save_results, 'voxels_dist_v2.png'), bbox_inches='tight')


#---------- run functions
# Assume that the dataset has pass the first step of preprocessing.
# Meaning that all the important files, are saved as .nii images.
#calculate_distribution_voxelsizes(path_to_dataset_updated, path_to_inter_results, round_decimal = 3)
calculate_distribution_resolution(path_to_dataset_updated, path_to_inter_results, round_decimal = 3)