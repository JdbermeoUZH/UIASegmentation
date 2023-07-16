import os
import sys
sys.path.append('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation')

from general_utils import utils
from general_utils import MYParser
from models import model_utils as mu
from models import testing_utils as tu
from torch.utils.data import DataLoader
from dataloading import dataloaders as dl
from dataloading import datasets_v2 as datasets_v2


def get_testdataloader_v2(type_of_loader, 
                          data,
                          path_data, 
                          batch_size, 
                          shuffle,
                          num_workers,
                          config):
    
    transform = dl.get_transform_test(config)
    dataset   = datasets_v2.UIA_Dataset_v2(path_data, data, transform, config)
        
    custom_loader = DataLoader(dataset,
                               batch_size  = batch_size,
                               shuffle     = shuffle,
                               num_workers = num_workers,
                               pin_memory  = False)
    return custom_loader

def testing_interface(test_dataloader, config, param_dict, split_dict):
    
    if param_dict['ensemble']:
        
        print(f"Testing the {config.which_net} architecture using ensemble methods for task 2")
        
        model_dict               = dict()
        model_dict['model_name'] = 'ensemble_model'
        model_dict.update(param_dict)

        #---------- file paths creation
        path_to_save_test = model_dict['save_path'] + '/' +\
                            'v2_' + config.which_net + '_'+\
                            config.experiment_type  + '_' +\
                            config.exp_name + '_'         +\
                            model_dict['en_method']       +\
                            '_testing'
        
        if os.path.exists(path_to_save_test) == False:  os.makedirs(path_to_save_test)
        model_dict['path_to_save_test'] = path_to_save_test

        if model_dict['save_extend'] == True:
            #--- save the metrics for each test instance
            individual_scores_path = path_to_save_test + '/' + 'individual_scores.csv'
            model_dict['ind_scores_path'] = individual_scores_path

    
        tu.ensemble_model_predict_v2(test_dataloader,
                                     config,
                                     model_dict)


#---------- TESTING MODELS
save_extend  = True
save_path    = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/testing_results'

###---------- UNET
config       = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet_v2.py')
models_paths = ['binary_experiment_only_aneur_Jun24/only_vessels_500_b_0_binary_class_unet_baseline_1688472851/only_vessels_500_b_0_binary_class_unet_baseline_1688472851_244.mod',
                'binary_experiment_only_aneur_Jun24/only_vessels_500_b_1_binary_class_unet_baseline_1688472867/only_vessels_500_b_1_binary_class_unet_baseline_1688472867_201.mod',
                'binary_experiment_only_aneur_Jun24/only_vessels_500_b_2_binary_class_unet_baseline_1688472901/only_vessels_500_b_2_binary_class_unet_baseline_1688472901_283.mod']

config       = config.config_namespace
models_paths = [config.path_to_models + '/' +i for i in models_paths]

split_dict   = utils.load_data(config.path_data,
                               config.path_splits,
                               0,
                               config.train_data_name,
                               config.valid_data_name,
                               config.test_data_name)

test_dataloader = get_testdataloader_v2('test',
                                        split_dict['test'],
                                        config.path_data,
                                        config.batch_size_test,
                                        config.shuffle_test,
                                        config.num_workers,
                                        config
                                        )

param_dict                 = dict()
param_dict['ensemble']     = True
param_dict['en_method']    = 'voting' # or 'mean_aggr'
param_dict['exp_name']     = config.exp_name
param_dict['exp_type']     = config.experiment_type
param_dict['models']       = models_paths
param_dict['save_path']    = save_path
param_dict['save_extend']  = save_extend

if os.path.exists(param_dict['save_path']) == False:    
    os.makedirs(param_dict['save_path'])

testing_interface(test_dataloader, config, param_dict, split_dict)
