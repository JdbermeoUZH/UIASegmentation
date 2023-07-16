'''
This script loads the 5 models trained for the binary problem,
which achieve good segmentation masks. It computes the predicted
segmentation masks for all the images and saves the in h5 format.
The script saves as 
_init.h5py the segmentation mask with the real values
_mask.h5py the mask with 0-1 and 
_segm.h5py the real segmentation mask
'''

import os
import sys
import h5py
import torch
import numpy as np
from tqdm import tqdm
sys.path.append('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation')


from general_utils import utils
from general_utils import MYParser
from models import model_utils as mu
from models import testing_utils as tu

def save_hdf5_img(img_data, img_aff, img_header, path_to_save):
    with h5py.File(path_to_save, 'w') as f:
        dset = f.create_dataset('data', data=img_data)
        # maybe useless
        #dset.attrs['affine'] = img_aff

def create_new_dataset(name, pred_segm_mask, segm_image2, image, path):
    
    if os.path.exists(path) == False:   os.makedirs(path)

    pred_image      = torch.mul(image, pred_segm_mask)

    pred_image_np   = pred_image.detach().cpu().numpy()
    pred_image_np   = np.squeeze(pred_image_np, axis = (0, 1))
    pred_image_path = path + '/' + name.lower() + '_new_tof.h5'

    pred_mask_np    = pred_segm_mask.detach().cpu().numpy()
    pred_mask_np    = np.squeeze(pred_mask_np, axis = (0, 1))
    pred_mask_path  = path + '/' + name.lower() + '_new_mask.h5'

    gt_segm_mask_np = segm_image2.detach().cpu().numpy()
    gt_segm_mask_np = np.squeeze(gt_segm_mask_np, axis = (0, 1))
    gt_segm_path    = path + '/' + name.lower() + '_seg.h5'

    save_hdf5_img(pred_image_np, None, None, pred_image_path)
    save_hdf5_img(pred_mask_np, None, None, pred_mask_path)
    save_hdf5_img(gt_segm_mask_np, None, None, gt_segm_path)

folder_name          = 'hdf5_dataset_only_vessels' 
save_path            = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/{folder_name}'
path_to_save_metrics = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/results/new_dataset_metrics'
path_to_ind_scores   = path_to_save_metrics + '/individual_scores.csv'

config       = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet.py')
models_paths = ['binary_experiment_Jun6/unet_binary/f0_lr001_binary_class_unet_baseline_1686006234/f0_lr001_binary_class_unet_baseline_1686006234_44.mod',
                'binary_experiment_Jun6/unet_binary/f1_lr001_binary_class_unet_baseline_1686006234/f1_lr001_binary_class_unet_baseline_1686006234_35.mod',
                'binary_experiment_Jun6/unet_binary/f2_lr001_binary_class_unet_baseline_1686006234/f2_lr001_binary_class_unet_baseline_1686006234_47.mod',
                'binary_experiment_Jun6/unet_binary/f3_lr001_binary_class_unet_baseline_1686006252/f3_lr001_binary_class_unet_baseline_1686006252_48.mod',
                'binary_experiment_Jun6/unet_binary/f4_lr001_binary_class_unet_baseline_1686006276/f4_lr001_binary_class_unet_baseline_1686006276_46.mod']

config       = config.config_namespace
models_paths = [config.path_to_models + '/' +i for i in models_paths]

split_dict   = utils.load_data(config.path_data,
                               config.path_splits,
                               0,
                               config.train_data_name,
                               config.valid_data_name,
                               config.test_data_name)
all_dataset = split_dict['train'] + split_dict['test'] + split_dict['valid']

test_dataloader = tu.get_testdataloader('test',
                                        all_dataset,
                                        config.path_data,
                                        config.batch_size_test,
                                        config.shuffle_test,
                                        config.num_workers,
                                        config
                                        )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models = list()

for model_path in models_paths:
    try:
        model = tu.load_model_weights(model_path, config, device)
    except Exception as e:
        print(f'ERROR: Could not load model from path: {model_path}... Exiting')
        print(e)
        raise RuntimeError
    models.append(model)
print("INFO: Models loading completed")


#---------- logs and metrics
eval_metrics   = mu.get_evaluation_metrics() 
test_collector = mu.MetricsCollector(eval_metrics, config, path_to_save_metrics)
test_epoch     = mu.MetricsClass(eval_metrics, config.experiment_type)

print("INFO: Testing-v3 started")
counter           = 0
individual_scores = dict()
aneurysm_scores   = dict()

with tqdm(test_dataloader, unit='batch') as tqdm_loader:
    for name, _, image, _, segm_image, segm_image2 in tqdm_loader:
        counter += 1
        name     = name[0]
        print(f'{counter}, predict for {name}')

        pred_image = None
        for model in models:
            pred_temp = tu.test_model_v3(model, image, segm_image, device)
            if pred_image == None:
                pred_image = mu.binarize_image(pred_temp)
            else:
                pred_image += mu.binarize_image(pred_temp)
        pred_image = torch.where(pred_image >= len(models)/2.0, 1, 0)
        
        
        test_epoch(pred_image, segm_image)
        individual_scores[name] = test_epoch.get_last_item()
        aneurysm_scores         = tu.calculate_aneurysm_metrics(pred_image, segm_image2)
        individual_scores[name]['recall_aneur'] = aneurysm_scores['recall_aneur']

        create_new_dataset(name, pred_image, segm_image2, image, save_path)

#---------- save results and metrics
test_epoch.print_aggregate_results()
test_collector.add_epoch(test_epoch, 0, 0, 0)
test_collector.save_config()
test_collector.save_logs()
tu.save_individual_metrics(individual_scores, path_to_ind_scores)  