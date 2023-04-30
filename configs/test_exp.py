#*********************************************
#*************** CONFIGURATION ***************
#*********************************************

# TEST CONFIGURATION

#---------- imports
import time
import torch


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


#---------- paths
path_data                 = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/{folder_name}'
path_splits               = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/datasets/{splits_name}'
path_results              = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/results/{experiment_name}'
path_intermediate_results = f'/scratch_net/biwidl210/kvergopoulos/SemesterProject/intermediate_results/{experiment_name}'


#---------- variables
shuffle_train      = True
shuffle_validation = False
shuffle_test       = False
device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers        = 1

#---------- Hyperparameters 
batch_size      = 1
batch_size_val  = 1
batch_size_test = 1
