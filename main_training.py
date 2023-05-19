# imports 
from models import train
from general_utils import utils
from general_utils import MYParser
from dataloading import dataloaders
from models import model_utils as mu

def main():

    # this part can be improved. For now it's ok
    config_file = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/graph_noskip.py'
    config      = MYParser.MyParser(config_file)
    config      = config.config_namespace

    #---------- initialize important variables
    device          = config.device
    experiment_name = config.experiment_name
    n_epochs        = config.number_of_epochs

    model           = mu.get_model(config.which_net, config)
    model.to(device)
    
    optimizer       = mu.get_optimizer(model, 
                                       config.which_optimizer, 
                                       config.learning_rate, 
                                       config.weight_decay)
    criterion       = mu.get_loss(config.which_loss)
    
    #---------- fetch the data
    split_dict = utils.load_split_dict(config.path_data, 
                                       config.path_splits, 
                                       config.fold_id, 
                                       config.train_data_name, 
                                       config.valid_data_name, 
                                       config.test_data_name)
    
    #---------- init dataloaders
    train_dataloader, val_dataloader, test_dataloader = dataloaders.get_dataloaders_all(config, split_dict)
    
    #---------- TRAIN MODEL
    if config.only_unets_flag == True:
        log = train.train_v1(model,
                             optimizer,
                             criterion,
                             n_epochs,
                             device,
                             config,
                             train_dataloader, 
                             val_dataloader,
                             test_dataloader, 
                             experiment_name)
    elif config.only_unets_flag == False:
        log = train.train_v2(model,
                             optimizer,
                             criterion,
                             n_epochs,
                             device,
                             config,
                             train_dataloader, 
                             val_dataloader,
                             test_dataloader, 
                             experiment_name)
        
    return None

# entry point
if __name__ == '__main__':    
    main()