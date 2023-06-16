import numpy as np
import nibabel as nib
import preprocessing_utils as pu

# #---------- paths & hyperparameters
# hardcode them for now. Later maybe replace them.
path_to_data = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results/images_intermediate_folder/global_thresholded_99.5'

precision_list = list()
recall_list    = list()
dice_list      = list()
subjects_tofs_segm_dict = pu.locate_tof_mask_paths(path_to_data)
assert len(subjects_tofs_segm_dict) == 62

test_names = ["10941965_TSV_MCA_ICA",
              "10448252_BPE_ICA",
              "10739823-PComm-links-NEW",
              "10599468-AComm-new",
              "11137190_NM_MA_NEW",
              "11092603_GDSCR_MCA",
              "10854414_SS_MCA",
              "11140272_DSRMSM_ICA_NEW",
              "11102734_RD_MCA_AComm_NEW",
              "07767625_LJ_Pericallosa",
              "02014629_KO_MCA",
              "10299485_BMM_AComm_MCA"]

#for key in subjects_tofs_segm_dict:
for key in test_names:
    key = key.lower()
    img_thr       = nib.load(subjects_tofs_segm_dict[key]['mask'][0]).get_fdata()
    pred_mask_bin = pu.binarize(img_thr, 0) 
    img_seg       = nib.load(subjects_tofs_segm_dict[key]['seg'][0]).get_fdata()
    gt_mask_bin   = pu.binarize(img_seg, 0)
    assert pred_mask_bin.shape == gt_mask_bin.shape
    dice_list.append(pu.dice_metric(pred_mask_bin, gt_mask_bin))
    recall_list.append(pu.recall_metric(pred_mask_bin, gt_mask_bin))
    precision_list.append(pu.precision_metric(pred_mask_bin, gt_mask_bin))
print("Dice scores: ",  np.mean(dice_list), np.std(dice_list), np.min(dice_list), np.max(dice_list))
print("Recall scores: ",np.mean(recall_list), np.std(recall_list), np.min(recall_list), np.max(recall_list))
print("Precision scores: ",np.mean(precision_list), np.std(precision_list), np.min(precision_list), np.max(precision_list))

'''
recall_list_aneur    = list()
dice_list_aneur      = list()
precision_list_aneur = list()

for key in test_names:
    key           = key.lower()
    img_thr       = nib.load(subjects_tofs_segm_dict[key]['mask'][0]).get_fdata()
    pred_mask_bin = pu.binarize(img_thr, 0) 
    img_seg       = nib.load(subjects_tofs_segm_dict[key]['seg'][0]).get_fdata()
    aneur_mask_gt = np.where(img_seg == 4, 1, 0)
    assert pred_mask_bin.shape == aneur_mask_gt.shape
    pred_mask_aneur = np.multiply(pred_mask_bin, aneur_mask_gt) 
    
    dice_list_aneur.append(pu.dice_metric(pred_mask_aneur, aneur_mask_gt))
    recall_list_aneur.append(pu.recall_metric(pred_mask_aneur, aneur_mask_gt))
    precision_list_aneur.append(pu.precision_metric(pred_mask_aneur, aneur_mask_gt))
print("Dice scores: ",  np.mean(dice_list_aneur), np.std(dice_list_aneur), np.min(dice_list_aneur), np.max(dice_list_aneur))
print("Recall scores: ",np.mean(recall_list_aneur), np.std(recall_list_aneur), np.min(recall_list_aneur), np.max(recall_list_aneur))
print("Precision scores:", np.mean(precision_list_aneur), np.std(precision_list_aneur), np.min(precision_list_aneur), np.max(precision_list_aneur))
'''