import time
import torch
import numpy as np
from tqdm import tqdm
from models import model_utils as mu

def train_v3(model,
             optimizer,
             criterion,
             nepochs,
             device,
             config,
             train_dataloader = None, 
             valid_dataloader = None
             ):
    
    #---------- logs and metrics
    eval_metrics   = mu.get_evaluation_metrics() 
    eval_collector = mu.MetricsCollector(eval_metrics, config)
    eval_counter   = 0

    #---------- init steps
    saver     = mu.MSaver(config.path_to_models, config.experiment_name)

    scheduler = None
    if config.use_scheduler == True:
        scheduler = mu.LRScheduler(optimizer, train_dataloader, nepochs, config.which_scheduler, config)
    
    earlystopper = None
    if config.use_early_stopping == True:
        earlystopper = mu.EarlyStopping(patience = config.patience)    

    #---------- TRAINNING & VALIDATION
    for tepoch in range(nepochs):
        epoch_start_time   = time.time()
        print(f'{time.ctime(epoch_start_time)}: epoch: {tepoch}/{nepochs}')
        
        #---------- TRAINNING LOOP
        model.train()
        train_counter      = 0
        running_loss_train = 0.0
        # each batch contains n_images, n_patches, 1 channel, patch_size_x, patch_size_y, patch_size_z
        with tqdm(train_dataloader, unit='batch') as tqdm_loader:
            for _, image, _, segm_image in tqdm_loader:
                train_counter += 1
                image          = image.to(device)
                segm_image     = segm_image.to(device)
        
                optimizer.zero_grad()
                image      = model(image)
                loss_train = criterion(image, segm_image)
                loss_train.backward()
                optimizer.step()
                running_loss_train   += loss_train.item()

                del loss_train, image, segm_image
        #---------- print out message
        train_end_time   = time.time()
        print(f'TRAINING: {time.ctime(train_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_train/train_counter}')
        #---------- 
        
        #---------- VALIDATION LOOP
        model.eval()
        with torch.no_grad():
            valid_counter    = 0
            running_loss_val = 0.0
            eval_epoch       = mu.MetricsClass(eval_metrics, config.experiment_type)
            # each batch contains n_images, n_patches, 1 channel, patch_size_x, patch_size_y, patch_size_z
            with tqdm(valid_dataloader, unit='batch') as tqdm_loader:
                for _, image, _, segm_image in tqdm_loader:
                    valid_counter    += 1
                    image             = image.to(device)
                    segm_image        = segm_image.to(device)
                    image             = model(image)
                    loss_val          = criterion(image, segm_image)
                    running_loss_val += loss_val.item()
                    
                    # compute evaluation metrics
                    eval_epoch(image, segm_image)
                    
                    del loss_val, image, segm_image
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        eval_epoch.print_aggregate_results()
        eval_collector.add_epoch(eval_epoch, tepoch, running_loss_train/train_counter, running_loss_val/valid_counter)
        #----------

        #---------- update parameters
        eval_counter += 1
        if eval_counter > 15:
            eval_counter = 0
            eval_collector.save_config()
            eval_collector.save_logs()
            eval_collector.save_training_figs()

        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'), tepoch)
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            print(f'INFO: Early stopping {tepoch}')
            break

    return eval_collector