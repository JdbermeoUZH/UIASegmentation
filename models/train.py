import time
import torch
from tqdm import tqdm

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
    pass

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
            for names, adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                
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
            with tqdm(valid_dataloader, unit='batch') as tqdm_loader:
                for names, adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                    
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
                    loss_val    = criterion(batch_preds, node_fts_gt)
                    running_loss_val += loss_val.item()
        #---------- print out message
        valid_end_time   = time.time()
        print(f'VALIDATION: {time.ctime(valid_end_time)}: epoch: {tepoch}/{nepochs} loss: {running_loss_val/valid_counter}')
        #----------