# imports 
import matplotlib.pyplot as plt
from general_utils import utils
from general_utils import MYParser
from dataloading import dataloaders

def train(config):

    #---------- initialize important variables
    device          = config.device
    experiment_name = config.experiment_name

    #---------- data loaders
    split_dict = utils.load_split_dict(config.path_data, 
                                       config.path_splits, 
                                       config.fold_id, 
                                       config.train_data_name, 
                                       config.valid_data_name, 
                                       config.test_data_name)
    
    train_dataloader, val_dataloader, test_dataloader = dataloaders.get_dataloaders_all(config, split_dict)
    
    # test dataloader
    #for batch_ndx, sample in enumerate(train_dataloader):
    #    print(batch_ndx)
    #    print(sample[0], sample[1].is_pinned(), sample[2].is_pinned(), sample[3].is_pinned())

    source_iter = iter(train_dataloader)
    names, images, masks, segmentations = next(source_iter)
    #print(images.shape)
    #names, images, masks, segmentations = next(source_iter)
    
    return None

# entry point
if __name__ == '__main__':  

    # this part can be improved. For now it's ok
    config_file = '/scratch_net/biwidl210/kvergopoulos/SemesterProject/UIASegmentation/configs/test_exp_cpu.py'
    config      = MYParser.MyParser(config_file)

    logs        = train(config.config_namespace)
    print("Training ends !!!") 