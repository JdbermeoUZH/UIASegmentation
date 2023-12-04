#*********************************************
#*************** CONFIGURATION ***************
#*********************************************
# UNET for the whole volume

#---------- imports
import time
import torch


#---------- names
which_net       = 'unet_baseline'  # UNet3D with skip connections and (1, 2, 4, 16, 32) filters

# The .json file contains a dict with keys in format *_data. 
# Each key corresponds to training, validation and test set
splits_name     = 'usz_kfold_5.json'
train_data_name = 'train_data'
valid_data_name = 'valid_data'
test_data_name  = 'test_data'
fold_id         = 0

# folder name of the dataset
folder_name     = 'hdf5_dataset_sample'


#---------- experiments
exp_name        = 'nodiceforback_f0_50'

experiment_type = 'binary_class_all_vessels'    # for all vessels
#experiment_type = 'binary_class'               # for aneurysm only
#experiment_type = 'three_class'                # for background, vessels, aneurysm
#experiment_type = 'multi_class'                # for all classes 

timestamp       = int(time.time())
experiment_name = f'{exp_name}_{experiment_type}_{which_net}_{timestamp}'


#---------- paths
path_data                 = f"/home/juanbermeo/Documents/Master's Thesis/data/preprocessed/kostas/{folder_name}"
path_splits               = f"/home/juanbermeo/Documents/Master's Thesis/UIASegmentation/configs/{splits_name}"
path_results              = f"/home/juanbermeo/Documents/Master's Thesis/UIASegmentation/configs/results"
path_to_models            = f"/home/juanbermeo/Documents/Master's Thesis/UIASegmentation/configs/results/models"


#---------- variables
shuffle_train      = True
shuffle_validation = True
shuffle_test       = False
device             = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_workers        = 2

#---------- Hyperparameters 
batch_size         = 1
batch_size_val     = 1
batch_size_test    = 1
# the following are not used.
graph_patch_size   = [32,32,16]
graph_connectivity = 26 # available options 6 (only faces), 18(faces & edges), 26(faces, edges & corners)

#---------- data processing
normalization                = 'min_max' # choose between 'min_max' or 'standardization'
max_normalization_percent    = 99.8
min_normalization_percent    = 0.5
transforms_probability       = 0.5
transforms_probability_valid = 0 

#---------- model 
only_unets_flag     = False
use_patches         = False
use_gae             = False
number_of_epochs    = 50
use_early_stopping  = True
patience            = 20
activation_function = 'relu'

output_channels     = 1
#output_channels     = 3
#output_channels     = 22

which_optimizer     = 'adam'
learning_rate       = 0.001
use_scheduler       = False
which_scheduler     = 'r' # options are: 'reduce_lr_on_plateau', 'one_cylce_lr', 'step_lr'
weight_decay        = 0.001

which_loss          = 'dice_loss'
#which_loss          = 'multidice_loss'