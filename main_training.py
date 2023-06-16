# imports 
import random
import torch
import numpy as np
from models import train
from general_utils import utils
from general_utils import MYParser
from dataloading import dataloaders
from models import model_utils as mu

def seed_all_rng(seed):
    # Set seed for Python's built-in random number generator
    random.seed(seed)

    # Set seed for NumPy's random number generator
    np.random.seed(seed)

    # Set seed for PyTorch's random number generator
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.cuda.manual_seed_all(seed)

def main():

    #seed_all_rng(123)
    
    config      = MYParser.MyParser()
    # --- remove afrer implementation
    #config      = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet.py')
    #config      = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet_with_gae.py')
    #config      = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/gae_v2.py')
    # end remove ---
     
    config      = config.config_namespace

    #---------- initialize important variables
    device          = config.device
    n_epochs        = config.number_of_epochs

    model           = mu.get_model(config.which_net, config)
    model.to(device)
    
    optimizer       = mu.get_optimizer(model, 
                                       config.which_optimizer, 
                                       config.learning_rate, 
                                       config.weight_decay)
    criterion       = mu.get_loss(config.which_loss)
    
    #---------- fetch the data
    split_dict = utils.load_data(config.path_data, 
                                 config.path_splits, 
                                 config.fold_id,
                                 config.train_data_name,
                                 config.valid_data_name,
                                 config.test_data_name)
    
    #---------- init dataloaders
    train_dataloader, val_dataloader, _ = dataloaders.get_dataloaders_all(config, split_dict)
    
    #---------- TRAIN MODEL
    # because the architectures have diferences,
    # there are diferent functions to properly train the model 
    if config.which_net == 'unet_no_skip_connections' or config.which_net == 'unet_skip_connections':
        print("INFO: Train-v1 started")
        log = train.train_v1(model,
                            optimizer,
                            criterion,
                            n_epochs,
                            device,
                            config,
                            train_dataloader, 
                            val_dataloader)
    elif config.which_net == 'combnet_v1' or config.which_net == 'combnet_v2':
        print("INFO: Train-v2 started")
        log = train.train_v2(model,
                            optimizer,
                            criterion,
                            n_epochs,
                            device,
                            config,
                            train_dataloader, 
                            val_dataloader)
    elif config.which_net == 'combnet_v3' or config.which_net == 'combnet_v4':
        print("INFO: Train-v4 started")
        log = train.train_v4(model,
                            optimizer,
                            criterion,
                            n_epochs,
                            device,
                            config,
                            train_dataloader, 
                            val_dataloader)
    elif config.which_net == 'unet_baseline':
        print("INFO: train-v3 started")
        log = train.train_v3(model,
                             optimizer,
                             criterion,
                             n_epochs,
                             device,
                             config,
                             train_dataloader, 
                             val_dataloader)
    elif config.which_net == 'combnet_v5':
        print("INFO: train-v5 started")
        log = train.train_v5(model,
                             optimizer,
                             criterion,
                             n_epochs,
                             device,
                             config,
                             train_dataloader, 
                             val_dataloader)
    else:
        print(f"ERROR: there is no model of type:{config.which_net}")
        raise NameError
    
    # print messages save logs and plot some figs
    log.save_config()
    log.save_logs()
    log.save_training_figs()
    log.print_best_metrics()

    print("INFO: training ended... exiting")
    return

# entry point
if __name__ == '__main__':  
    '''
    RUN: python -c  config_file
    '''  
    main()