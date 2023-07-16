'''
The _v2 files are used for implementing a second approach. This implementation
is only for testing and also is optional. Hence, can be safely
ignored.
'''
from models import train_v2
from general_utils import utils
from general_utils import MYParser
from dataloading import dataloaders_v2 as dataloaders_v2
from models import model_utils as mu

def main():
    
    config      = MYParser.MyParser()
    #config      = MYParser.MyParser('/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/vanilla_unet_v2.py')
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
    train_dataloader, val_dataloader, _ = dataloaders_v2.get_dataloaders_all(config, split_dict)
    
    #---------- TRAIN MODEL 
    if config.which_net == 'unet_baseline':
        print("INFO: Train-v3 started")
        log = train_v2.train_v3(model,
                                optimizer,
                                criterion,
                                n_epochs,
                                device,
                                config,
                                train_dataloader, 
                                val_dataloader)
        
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