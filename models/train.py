import time
import torch
import numpy as np
from tqdm import tqdm
from models import model_utils as mu


# this train is only used for models with only unets blocks
def train_v1(model,
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

                    # compute evaluation metrics
                    eval_epoch(batch_preds, node_fts_gt)

                    del loss_val, node_fts, node_fts_gt, adj_mtx, adj_mtx_gt, batch_preds
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        eval_epoch.print_aggregate_results()
        eval_collector.add_epoch(eval_epoch, tepoch, running_loss_train/train_counter, running_loss_val/valid_counter)
        #----------

        #---------- update parameters
        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'), tepoch)
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            print(f'INFO: Early stopping {tepoch}')
            break
        
    return eval_collector

def train_v2(model,
             optimizer,
             criterion,
             nepochs,
             device,
             config,
             train_dataloader = None, 
             valid_dataloader = None
             ):
    #--- only for debug
    torch.autograd.set_detect_anomaly(True)
    # only for debug --- 

    #---------- logs and metrics
    clip_value     = 2
    eval_metrics   = mu.get_evaluation_metrics() 
    eval_collector = mu.MetricsCollector(eval_metrics, config)

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
            for adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                train_counter += 1
                node_fts       = node_fts.to(device)
                adj_mtx        = adj_mtx.to(device)
                node_fts_gt    = node_fts_gt.to(device)
                adj_mtx_gt     = adj_mtx_gt.to(device)
                assert node_fts.shape == node_fts_gt.shape, f'Training: got different dimensions for preds:{node_fts.shape} and training: {node_fts_gt.shape}'
        
                optimizer.zero_grad()
                node_fts_preds, adj_mtx_preds = model(node_fts, adj_mtx)
                loss_train                    = criterion(node_fts_preds, 
                                                          node_fts_gt, debug_mode=False)        
                loss_train.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
                running_loss_train   += loss_train.item()
                del loss_train, node_fts, node_fts_gt, adj_mtx, adj_mtx_gt, node_fts_preds, adj_mtx_preds
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
                    
                    node_fts_preds, adj_mtx_preds = model(node_fts, adj_mtx)
                    loss_val                      = criterion(node_fts_preds, 
                                                              node_fts_gt)
                    running_loss_val             += loss_val.item()
                    
                    # compute evaluation metrics
                    eval_epoch(node_fts_preds, node_fts_gt)

                    del loss_val, node_fts, node_fts_gt, adj_mtx, adj_mtx_gt, node_fts_preds, adj_mtx_preds
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        eval_epoch.print_aggregate_results()
        eval_collector.add_epoch(eval_epoch, tepoch, running_loss_train/train_counter, running_loss_val/valid_counter)
        #----------

        #---------- update parameters
        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'), tepoch)
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            print(f'INFO: Early stopping {tepoch}')
            break
    
    return eval_collector

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
        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'), tepoch)
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            print(f'INFO: Early stopping {tepoch}')
            break

    return eval_collector

def train_v4(model,
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
            for adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                train_counter += 1
                node_fts       = node_fts.to(device)
                adj_mtx        = adj_mtx.to(device)
                node_fts_gt    = node_fts_gt.to(device)
                adj_mtx_gt     = adj_mtx_gt.to(device)
        
                optimizer.zero_grad()
                node_fts_preds, adj_mtx_preds, adj_mtx_weights_preds = model(node_fts, adj_mtx)
                loss_train = criterion(node_fts_preds,
                                       node_fts_gt,
                                       adj_mtx_preds,
                                       adj_mtx_weights_preds,
                                       adj_mtx_gt)
                loss_train.backward()
                optimizer.step()
                running_loss_train   += loss_train.item()
                del loss_train, node_fts, node_fts_preds, node_fts_gt, adj_mtx, adj_mtx_preds, adj_mtx_weights_preds, adj_mtx_gt
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
                for adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                    valid_counter += 1
                    node_fts       = node_fts.to(device)
                    adj_mtx        = adj_mtx.to(device)
                    node_fts_gt    = node_fts_gt.to(device)
                    adj_mtx_gt     = adj_mtx_gt.to(device)
                    
                    node_fts_preds, adj_mtx_preds, adj_mtx_weights_preds = model(node_fts, adj_mtx)
                    loss_val = criterion(node_fts_preds,
                                         node_fts_gt,
                                         adj_mtx_preds,
                                         adj_mtx_weights_preds,
                                         adj_mtx_gt)
                    running_loss_val += loss_val.item()
                    
                    # compute evaluation metrics
                    eval_epoch(node_fts_preds, node_fts_gt)
                    
                    del loss_val, node_fts, node_fts_preds, node_fts_gt, adj_mtx, adj_mtx_preds, adj_mtx_weights_preds, adj_mtx_gt
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        eval_epoch.print_aggregate_results()
        eval_collector.add_epoch(eval_epoch, tepoch, running_loss_train/train_counter, running_loss_val/valid_counter)
        #----------

        #---------- update parameters
        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'), tepoch)
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            print(f'INFO: Early stopping {tepoch}')
            break
    
    return eval_collector

def train_v5(model,
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
            for adj_mtx, image, adj_mtx_gt, segm_image in tqdm_loader:
                train_counter += 1
                adj_mtx        = adj_mtx.to(device)
                image          = image.to(device)
                adj_mtx_gt     = adj_mtx_gt.to(device)
                segm_image     = segm_image.to(device)
        
                optimizer.zero_grad()
                image, adj_mtx_preds, adj_mtx_weights_preds = model(image, adj_mtx)
                
                loss_train = criterion(image, 
                                       segm_image,
                                       adj_mtx_preds, 
                                       adj_mtx_weights_preds,
                                       adj_mtx_gt)
                
                loss_train.backward()
                optimizer.step()
                running_loss_train   += loss_train.item()

                del loss_train, image, segm_image, adj_mtx, adj_mtx_preds, adj_mtx_weights_preds, adj_mtx_gt
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
                for adj_mtx, image, adj_mtx_gt, segm_image in tqdm_loader:
                    valid_counter += 1
                    adj_mtx        = adj_mtx.to(device)
                    image          = image.to(device)
                    adj_mtx_gt     = adj_mtx_gt.to(device)
                    segm_image     = segm_image.to(device)
                    
                    image, adj_mtx_preds, adj_mtx_weights_preds = model(image, adj_mtx)

                    loss_val = criterion(image, 
                                         segm_image,
                                         adj_mtx_preds, 
                                         adj_mtx_weights_preds,
                                         adj_mtx_gt)
                    
                    running_loss_val += loss_val.item()
                    
                    # compute evaluation metrics
                    eval_epoch(image, segm_image)
                    
                    del loss_val, image, segm_image, adj_mtx, adj_mtx_preds, adj_mtx_weights_preds, adj_mtx_gt
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        eval_epoch.print_aggregate_results()
        eval_collector.add_epoch(eval_epoch, tepoch, running_loss_train/train_counter, running_loss_val/valid_counter)
        #----------

        #---------- update parameters
        if scheduler != None: scheduler(running_loss_val)

        saver(model, eval_epoch.get_metric('dice_score'), tepoch)
        stop_flag = False
        if earlystopper != None:
            stop_flag = earlystopper(running_loss_val, tepoch)
        if stop_flag == True:
            print(f'INFO: Early stopping {tepoch}')
            break

    return eval_collector