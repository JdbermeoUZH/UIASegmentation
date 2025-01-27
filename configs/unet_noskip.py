#*********************************************
#*************** CONFIGURATION ***************
#*********************************************


#---------- imports
import time
import torch


#---------- names
which_net           = 'unet_no_skip_connections' # UNet3D without skip connections and (1, 2, 4, 8, 16) filters
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
exp_name        = 'first_exp'
experiment_type = 'binary_class' # choose between 'binary_class' or 'three_class' or 'multi_class' 
timestamp       = int(time.time())
experiment_name = f'{exp_name}_{experiment_type}_{which_net}_{timestamp}'

#---------- paths
path_data                 = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/{folder_name}'
path_splits               = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/{splits_name}'
path_results              = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/results/{experiment_name}'
path_to_models            = f'/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/models'


#---------- variables
shuffle_train      = True
shuffle_validation = True
shuffle_test       = False
device             = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_workers        = 1

#---------- Hyperparameters 
batch_size         = 1
batch_size_val     = 1
batch_size_test    = 1
graph_patch_size   = [32,32,16]
graph_connectivity = 6 # available options 6 (only faces), 18(faces & edges), 26(faces, edges & corners)

#---------- data processing
normalization                = 'min_max' # choose between 'min_max' or 'standardization'
max_normalization_percent    = 99.8
min_normalization_percent    = 0.5
transforms_probability       = 0.5
transforms_probability_valid = 0 

#---------- model 
only_unets_flag     = True
use_patches         = True 
use_early_stopping  = False
use_scheduler       = False
number_of_epochs    = 20
output_channels     = 1
activation_function = 'relu'
which_optimizer     = 'adam'
learning_rate       = 0.001
weight_decay        = 0.001
which_loss          = 'dice_loss'