import os
import sys
sys.path.append('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation')

from general_utils import utils
from general_utils import MYParser
from models import model_utils as mu
from models import testing_utils as tu

def testing_interface(test_dataloader, config, param_dict):

    if param_dict['ensemble'] == False:
        for model_path in param_dict['models']:
            
            model_name = model_path.split('/')[-1].split('.')[0]            
            print(f"Testing {config.which_net} architecture and the model {model_name}")

            model_dict               = dict()
            model_dict['model_path'] = model_path
            model_dict['model_name'] = model_name
            model_dict.update(param_dict)
            
            
            #---------- file paths creation
            path_to_save_test = model_dict['save_path'] + '/' + model_name + '_testing'
            if os.path.exists(path_to_save_test) == False:  os.makedirs(path_to_save_test)
            model_dict['path_to_save_test'] = path_to_save_test

            if model_dict['save_extend'] == True:
                #--- save the resulted images
                save_images_path = path_to_save_test + '/' + 'npy_images'
                if os.path.exists(save_images_path) == False:  os.makedirs(save_images_path)
                model_dict['images_path'] = save_images_path

                #--- save the metrics for each test instance
                individual_scores_path = path_to_save_test + '/' + 'individual_scores.csv'
                model_dict['ind_scores_path'] = individual_scores_path
            
            #---------- testing
            tu.model_predict_single(test_dataloader, 
                                    config, 
                                    model_dict)
            del model_dict
    
    elif param_dict['ensemble'] == True:
        print(f"Testing the {config.which_net} architecture using ensemble methods")
        
        model_dict               = dict()
        model_dict['model_name'] = 'ensemble_model'
        model_dict.update(param_dict)

        #---------- file paths creation
        path_to_save_test = model_dict['save_path'] + '/' +\
                            config.which_net + '_'        +\
                            config.experiment_type  + '_' +\
                            model_dict['en_method']       +\
                            '_testing'
        
        if os.path.exists(path_to_save_test) == False:  os.makedirs(path_to_save_test)
        model_dict['path_to_save_test'] = path_to_save_test

        if model_dict['save_extend'] == True:
                #--- save the resulted images
                save_images_path = path_to_save_test + '/' + 'npy_images'
                if os.path.exists(save_images_path) == False:  os.makedirs(save_images_path)
                model_dict['images_path'] = save_images_path

                #--- save the metrics for each test instance
                individual_scores_path = path_to_save_test + '/' + 'individual_scores.csv'
                model_dict['ind_scores_path'] = individual_scores_path
        
        tu.ensemble_model_predict(test_dataloader,
                                  config,
                                  model_dict)



#---------- TESTING MODELS
save_extend  = True
save_path    = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/testing_results'

##---------- BINARY PROBLEM
###---------- WHOLE UNET
config       = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet.py')
models_paths = ['binary_models/f0_lr001_binary_class_unet_baseline_1686006234/f0_lr001_binary_class_unet_baseline_1686006234_44.mod',
                'binary_models/f1_lr001_binary_class_unet_baseline_1686006234/f1_lr001_binary_class_unet_baseline_1686006234_35.mod',
                'binary_models/f2_lr001_binary_class_unet_baseline_1686006234/f2_lr001_binary_class_unet_baseline_1686006234_47.mod',
                'binary_models/f3_lr001_binary_class_unet_baseline_1686006252/f3_lr001_binary_class_unet_baseline_1686006252_48.mod',
                'binary_models/f4_lr001_binary_class_unet_baseline_1686006276/f4_lr001_binary_class_unet_baseline_1686006276_46.mod',
                ]

###---------- WHOLE UNET WITH GAE
#config       = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet_with_gae.py')
#models_paths = ['/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/models/Jun_5/full_gae_lr0.001_dl1_gl1_binary_class_combnet_v5_1685909041/full_gae_lr0.001_dl1_gl1_binary_class_combnet_v5_1685909041_b.mod']

###---------- GAE
#config       = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/gae_v2.py')
#models_paths = ['/usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/USZ_BrainArtery/USZ_BrainArtery_GNN/models/Jun_5/gae_v2_lr0.001_dl1_gl1_binary_class_combnet_v4_1685909008/gae_v2_lr0.001_dl1_gl1_binary_class_combnet_v4_1685909008_b.mod']


config       = config.config_namespace
models_paths = [config.path_to_models + '/' +i for i in models_paths]

split_dict   = utils.load_data(config.path_data,
                               config.path_splits,
                               0,
                               config.train_data_name,
                               config.valid_data_name,
                               config.test_data_name)

test_dataloader = tu.get_testdataloader('test',
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
param_dict['exp_type']     = config.experiment_type
param_dict['models']       = models_paths
param_dict['save_path']    = save_path
param_dict['save_extend']  = save_extend

if os.path.exists(param_dict['save_path']) == False:    
    os.makedirs(param_dict['save_path'])

testing_interface(test_dataloader, config, param_dict)
