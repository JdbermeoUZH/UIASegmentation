import numpy as np
import nibabel as nib
import preprocessing_utils as pu

# #---------- paths & hyperparameters
# hardcode them for now. Later maybe replace them.
path_to_data = '/scratch/kvergopoulos/SemesterProject/datasets/pro_data_n4_False_sk_False_gl99.5'

recall_list = list()
dice_list   = list()
subjects_tofs_segm_dict = pu.locate_tof_mask_paths(path_to_data)
assert len(subjects_tofs_segm_dict) == 62

for key in subjects_tofs_segm_dict:
    img_thr       = nib.load(subjects_tofs_segm_dict[key]['tof'][0]).get_fdata()
    pred_mask_bin = pu.binarize(img_thr, 0) 
    img_seg       = nib.load(subjects_tofs_segm_dict[key]['seg'][0]).get_fdata()
    gt_mask_bin   = pu.binarize(img_seg, 0)
    dice_list.append(pu.dice_metric(pred_mask_bin, gt_mask_bin))
    recall_list.append(pu.recall_metric(pred_mask_bin, gt_mask_bin))
print("Dice scores: ",  np.mean(dice_list), np.std(dice_list), np.min(dice_list), np.max(dice_list))
print("Recall scores: ",np.mean(recall_list), np.std(recall_list), np.min(recall_list), np.max(recall_list))