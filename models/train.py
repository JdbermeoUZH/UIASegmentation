import time
import torch
from tqdm import tqdm

def train(model,
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
    
    #---------- TRAINNING
    for tepoch in range(nepochs):
        model.train()
        running_loss_train = 0.0
        epoch_start_time   = time.time()
        print(f'{time.ctime(epoch_start_time)}: epoch: {tepoch}/{nepochs}')
        
        with tqdm(train_dataloader, unit='batch') as tqdm_loader:
            # each batch contains n_images, n_patches, 1 channel, patch_size_x, patch_size_y, patch_size_z
            tt = -1
            for name, adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                tt += 1
                if tt > 0: break
                print(name, node_fts.shape, node_fts.requires_grad)
                node_fts.to(device)
                node_fts_gt.to(device)
                optimizer.zero_grad()
                
                batch_preds = []
                for image in range(node_fts.shape[0]):
                    i                = 0
                    minibatch        = 16
                    image_embeddings = []
                    while True:
                        index_start = i*minibatch
                        index_end   = (i+1)*minibatch
                        i          += 1
                        if index_end > node_fts.shape[1]:   index_end = node_fts.shape[1]
                        preds = model(node_fts[image, index_start:index_end, :, :, :, :])
                        image_embeddings.append(preds)
                        if i > 3: break
                    image_embeddings = torch.cat(image_embeddings, dim=0)
                    batch_preds.append(image_embeddings.view(1, *image_embeddings.shape))
                batch_preds = torch.cat(batch_preds, dim = 0)
                loss_train = criterion(batch_preds, node_fts_gt[:,0:64, :, :, :, :])
                loss_train.backward()
                print(batch_preds.grad)
                optimizer.step()