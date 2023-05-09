# imports 
from general_utils import utils
from general_utils import MYParser
from dataloading import dataloaders

def main():

    # this part can be improved. For now it's ok
    config_file = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/test_exp_cpu.py'
    config      = MYParser.MyParser(config_file)
    config      = config.config_namespace

    #---------- initialize important variables
    device          = config.device
    experiment_name = config.experiment_name

    #---------- fetch the data
    split_dict = utils.load_split_dict(config.path_data, 
                                       config.path_splits, 
                                       config.fold_id, 
                                       config.train_data_name, 
                                       config.valid_data_name, 
                                       config.test_data_name)
    
    #---------- init dataloaders
    train_dataloader, val_dataloader, test_dataloader = dataloaders.get_dataloaders_all(config, split_dict)

    #---------- init the model
    
    return None

# entry point
if __name__ == '__main__':  
    
    main()