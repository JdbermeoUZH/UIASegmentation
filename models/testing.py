from general_utils import utils
from general_utils import MYParser
from dataloading import dataloaders
from models import model_utils as mu

def testing_interface(test_dataloader, model_paths, config, ensemble = False):
    
    if ensemble == False:
        for model_path in model_paths:
            print(model_path)
            path_to_save_test_results = ''
            mu.model_predict_single(model_path, test_dataloader, config, path_to_save_test_results)
    elif ensemble == True:
        pass

# TESTING
config      = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet.py')
split_dict  = utils.load_data(config.path_data,
                              config.path_splits,
                              config.fold_id,
                              config.train_data_name,
                              config.valid_data_name,
                              config.test_data_name)
_, _, test_dataloader = dataloaders.get_dataloaders_all(config, split_dict)
ensemble              = False
model_paths           = ['path_to models_to_test']
testing_interface(test_dataloader, model_paths, config, ensemble)
