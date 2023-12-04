#*********************************************
#*************** CONFIGURATION ***************
#*********************************************


#---------- imports
import time
import torch


#---------- names
which_net       = 'unet_skip_connections' # UNet3D with skip connections and (1, 2, 4, 16, 32) filters

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
exp_name        = 'unet'
experiment_type = 'binary_class' # choose between 'binary_class' or 'three_class' or 'multi_class' 
timestamp       = int(time.time())
experiment_name = f'{exp_name}_{experiment_type}_{which_net}_{timestamp}'


#---------- paths
path_data                 = f'/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/USZ_BrainArtery/Processed/{folder_name}'
path_splits               = f'/scratch_net/biwidl319/jbermeo/UIASegmentation/configs/{splits_name}'
path_results              = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/results'
path_to_models            = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/results/models'



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
number_of_epochs    = 100
use_early_stopping  = True
patience            = 20
activation_function = 'relu'
output_channels     = 1
which_optimizer     = 'adam'
learning_rate       = 0.001
use_scheduler       = False
which_scheduler     = 'r' # options are: 'reduce_lr_on_plateau', 'one_cylce_lr', 'step_lr'
weight_decay        = 0.001
which_loss          = 'dice_loss'