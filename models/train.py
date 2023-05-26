import time
import torch
import numpy as np
from tqdm import tqdm
from models import model_utils as mu

def train_v2(model,
             optimizer,
             criterion,
             nepochs,
             device,
             config,
             train_dataloader = None, 
             valid_dataloader = None, 
             test_dataloader  = None,  
             exp_name         = ''
             ):
    
    #---------- logs and metrics
    eval_metrics   = mu.get_evaluation_metrics() 
    eval_collector = mu.MetricsCollector(eval_metrics, config)

    #---------- init steps
    saver     = mu.MSaver(config.path_to_models, config.which_net, config.experiment_type, config.exp_name)

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
            for adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                train_counter += 1
                node_fts       = node_fts.to(device)
                adj_mtx        = adj_mtx.to(device)
                node_fts_gt    = node_fts_gt.to(device)
                adj_mtx_gt     = adj_mtx_gt.to(device)
                assert node_fts.shape == node_fts_gt.shape, f'Training: got different dimensions for preds:{node_fts.shape} and training: {node_fts_gt.shape}'
        
                optimizer.zero_grad()
                node_preds, adj_preds = model(node_fts, adj_mtx)
                loss_train            = criterion(node_preds, node_fts_gt, adj_preds, adj_mtx_gt, adj_mtx)
                loss_train.backward()
                optimizer.step()
                running_loss_train   += loss_train.item()
                del loss_train, node_fts, node_fts_gt, adj_mtx, adj_mtx_gt, node_preds, adj_preds
        #---------- print out message
        train_end_time   = time.time()
        print(f'TRAINING: {time.ctime(train_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_train/train_counter}')
        #---------- 
        
        #---------- VALIDATION LOOP
        model.eval()
        with torch.no_grad():
            valid_counter    = 0
            running_loss_val = 0.0
            eval_epoch       = mu.MetricsClass(eval_metrics)
            # each batch contains n_images, n_patches, 1 channel, patch_size_x, patch_size_y, patch_size_z
            with tqdm(valid_dataloader, unit='batch') as tqdm_loader:
                for adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                    valid_counter += 1
                    node_fts       = node_fts.to(device)
                    adj_mtx        = adj_mtx.to(device)
                    node_fts_gt    = node_fts_gt.to(device)
                    adj_mtx_gt     = adj_mtx_gt.to(device)
                    assert node_fts.shape == node_fts_gt.shape, f'Validation: got different dimensions for preds:{node_fts.shape} and training: {node_fts_gt.shape}'
                    
                    node_preds, adj_preds = model(node_fts, adj_mtx)
                    loss_val              = criterion(node_preds, node_fts_gt, adj_preds, adj_mtx_gt, adj_mtx)
                    running_loss_val     += loss_val.item()
                    
                    # compute evaluation metrics
                    eval_epoch(node_preds, node_fts_gt, adj_preds, adj_mtx_gt)

                    del loss_val, node_fts, node_fts_gt, adj_mtx, adj_mtx_gt, node_preds, adj_preds
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        eval_epoch.print_aggregate_results()
        eval_collector.add_epoch(eval_epoch, tepoch, running_loss_train/train_counter, running_loss_val/valid_counter)
        #----------

        #---------- update parameters
        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'))
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            break
    
    return eval_collector

# this train is only used for models with only unets blocks
def train_v1(model,
             optimizer,
             criterion,
             nepochs,
             device,
             config,
             train_dataloader = None, 
             valid_dataloader = None, 
             test_dataloader  = None,  
             exp_name         = ''
             ):
    
    #---------- logs and metrics
    eval_metrics   = mu.get_evaluation_metrics() 
    eval_collector = mu.MetricsCollector(eval_metrics, config)

    #---------- init steps
    saver     = mu.MSaver(config.path_to_models, config.which_net, config.experiment_type, config.exp_name)

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
            for adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                train_counter += 1
                node_fts       = node_fts.to(device)
                node_fts_gt    = node_fts_gt.to(device)
                assert node_fts.shape == node_fts_gt.shape, f'Training: got different dimensions for preds:{node_fts.shape} and training: {node_fts_gt.shape}'
                batch_shape = node_fts.shape

                optimizer.zero_grad()
                node_fts    = node_fts.view(batch_shape[0]*batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
                batch_preds = []
                minibatch   = 256
                idx         = 0
                # pass all patches through the unet
                while True:
                    index_start = idx * minibatch
                    index_end   = (idx + 1) * minibatch
                    idx        += 1
                    if index_start >= node_fts.shape[0]: break
                    if index_end > node_fts.shape[0]:   index_end = node_fts.shape[0]
                    preds = model(node_fts[index_start:index_end, :, :, :, :])
                    batch_preds.append(preds)
                batch_preds = torch.cat(batch_preds, dim = 0)
                batch_preds = batch_preds.view(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
                loss_train  = criterion(batch_preds, node_fts_gt)
                loss_train.backward()
                optimizer.step()
                running_loss_train += loss_train.item()
                if train_counter > 3: break
                del loss_train, node_fts, node_fts_gt, adj_mtx, adj_mtx_gt, batch_preds
        #---------- print out message
        train_end_time   = time.time()
        print(f'TRAINING: {time.ctime(train_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_train/train_counter}')
        #---------- 
        
        #---------- VALIDATION LOOP
        model.eval()
        with torch.no_grad():
            valid_counter    = 0
            running_loss_val = 0.0
            dice_scores_val  = []
            eval_epoch       = mu.MetricsClass(eval_metrics)
            # each batch contains n_images, n_patches, 1 channel, patch_size_x, patch_size_y, patch_size_z
            with tqdm(valid_dataloader, unit='batch') as tqdm_loader:
                for adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                    
                    valid_counter += 1
                    node_fts       = node_fts.to(device)
                    node_fts_gt    = node_fts_gt.to(device)
                    assert node_fts.shape == node_fts_gt.shape, f'Validation: got different dimensions for preds:{node_fts.shape} and training: {node_fts_gt.shape}'
                    batch_shape = node_fts.shape
                    
                    node_fts    = node_fts.view(batch_shape[0]*batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
                    batch_preds = []
                    minibatch   = 256
                    idx         = 0
                    # pass all patches through the unet
                    while True:
                        index_start = idx * minibatch
                        index_end   = (idx + 1) * minibatch
                        idx        += 1
                        if index_start >= node_fts.shape[0]: break
                        if index_end > node_fts.shape[0]:   index_end = node_fts.shape[0]
                        preds = model(node_fts[index_start:index_end, :, :, :, :])
                        batch_preds.append(preds)
                    batch_preds = torch.cat(batch_preds, dim = 0)
                    batch_preds = batch_preds.view(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3], batch_shape[4], batch_shape[5])
                    
                    loss_val          = criterion(batch_preds, node_fts_gt)
                    running_loss_val += loss_val.item()
                    
                    # only for debug
                    dice_score        = mu.dice_score_metric(batch_preds, node_fts_gt)
                    dice_scores_val += dice_score
                    #

                    # compute evaluation metrics
                    eval_epoch(batch_preds, node_fts_gt, None, None)

                    del loss_val, node_fts, node_fts_gt, adj_mtx, adj_mtx_gt, batch_preds, dice_score
        #---------- print out message, debug
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}, dice score: {np.mean(dice_scores_val)}')
        #---------- delete afterwards
        
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        eval_epoch.print_aggregate_results()
        eval_collector.add_epoch(eval_epoch, tepoch, running_loss_train/train_counter, running_loss_val/valid_counter)
        #----------

        #---------- update parameters
        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'))
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            break
    return eval_collector
