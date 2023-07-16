#*********************************************
#*************** CONFIGURATION ***************
#*********************************************


#---------- imports
import time
import torch


#---------- names
which_net       = 'combnet_v3' 

# The .json file contains a dict with keys in format *_data. 
# Each key corresponds to training, validation and test set
splits_name     = 'usz_kfold_5.json'
train_data_name = 'train_data'
valid_data_name = 'valid_data'
test_data_name  = 'test_data'
fold_id         = 0

# folder name of the dataset
folder_name     = 'hdf5_dataset'


#---------- experiment
exp_name        = 'gaenet_v1_lr0.001_dec0_1_1'
experiment_type = 'binary_class' # choose between 'binary_class' or 'three_class' or 'multi_class' 
timestamp       = int(time.time())
experiment_name = f'{exp_name}_{experiment_type}_{which_net}_{timestamp}'


#---------- paths
path_data                 = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/{folder_name}'
path_splits               = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/{splits_name}'
path_results              = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/results'
path_to_models            = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/models'


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
transforms_probability       = 0.4
transforms_probability_valid = 0 

#---------- model 
only_unets_flag       = False
use_patches           = True
number_of_epochs      = 50
use_early_stopping    = True
patience              = 20
activation_function   = 'relu'
activation_function_g = 'relu'
depth_g_enc           = 2
depth_g_dec           = 0
hidden_channels_g     = 512 #check values 128, 256, 512, 1024
all_edges_g           = False
output_channels       = 1
which_optimizer       = 'adam'
learning_rate         = 0.001
use_scheduler         = False
which_scheduler       = 'r' # options are: 'reduce_lr_on_plateau', 'one_cylce_lr', 'step_lr'
weight_decay          = 0.001
which_loss            = 'graph_loss'