#*********************************************
#*************** CONFIGURATION ***************
#*********************************************


#---------- imports
import time
import torch


#---------- names
which_net       = 'combnet_v2' 

# The .json file contains a dict with keys in format *_data. 
# Each key corresponds to training, validation and test set
splits_name     = 'usz_kfold_5.json'
train_data_name = 'train_data'
valid_data_name = 'valid_data'
test_data_name  = 'test_data'
fold_id         = 0

# folder name of the dataset
folder_name     = 'hdf5_dataset_small'


#---------- experiment
exp_name        = 'first_exp'
experiment_type = 'binary_class' # choose between 'binary_class' or 'three_class' or 'multi_class' 
timestamp       = int(time.time())
experiment_name = f'{experiment_type}_{which_net}_{timestamp}'


#---------- paths
path_data                 = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/{folder_name}'
path_splits               = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/{splits_name}'
path_results              = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/results'
path_to_models            = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/models'
path_intermediate_results = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results/{experiment_name}'


#---------- variables
shuffle_train      = True
shuffle_validation = True
shuffle_test       = False
device             = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_workers        = 2

#---------- Hyperparameters 
batch_size         = 1
batch_size_val     = 1
batch_size_test    = 1
graph_patch_size   = [32,32,16]
graph_connectivity = 26 # available options 6 (only faces), 18(faces & edges), 26(faces, edges & corners)

#---------- data processing
normalization                = 'min_max' # choose between 'min_max' or 'standardization'
max_normalization_percent    = 99.8
min_normalization_percent    = 0.5
transforms_probability       = 0.5
transforms_probability_valid = 0 

#---------- model 
only_unets_flag       = False
number_of_epochs      = 1
use_early_stopping    = True
patience              = 10
activation_function   = 'relu'
activation_function_g = 'relu'
depth_g               = 3
hidden_channels_g     = 512 #check values 128, 256, 512, 1024
pool_ration_g         = 0.8
sum_res_g             = True
which_optimizer       = 'adam'
learning_rate         = 0.001
use_scheduler         = True
which_scheduler       = 'reduce_lr_on_plateau' # options are: 'reduce_lr_on_plateau', 'one_cylce_lr', 'step_lr'
weight_decay          = 0.001
which_loss            = 'graph_loss'