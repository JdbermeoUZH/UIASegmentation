#*********************************************
#*************** CONFIGURATION ***************
#*********************************************

# TEST CONFIGURATION

#---------- imports
import time
import torch


#---------- debugging
debug_mode = True

#---------- names
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
timestamp       = int(time.time())
experiment_name = f'{exp_name}_{timestamp}_{fold_id}'
experiment_type = 'binary_class' # choose between 'binary_class' or 'three_class' or 'multi_class' 

#---------- paths
path_data                 = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/{folder_name}'
path_splits               = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/{splits_name}'
path_results              = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/results/{experiment_name}'
path_intermediate_results = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results/{experiment_name}'


#---------- variables
shuffle_train      = True
shuffle_validation = True
shuffle_test       = False
device             = torch.device('cpu')
num_workers        = 1

#---------- Hyperparameters 
batch_size      = 1
batch_size_val  = 1
batch_size_test = 1

#---------- data processing
normalization             = 'min_max' # choose between 'min_max' or 'standardization'
max_normalization_percent = 99.8
min_normalization_percent = 0.5
transforms_probability    = 0.5