import torch
from models import uia_models as models

def get_model(which_model, config):
    '''
    This function initializes the model

    Parameters
    ----------
    which_model: The options are:
        1) Unet encoder/decoder without skip-connections
        2) ...

    Returns
    -------
    model: Torch neural network
    '''
    if which_model == 'unet_no_skip_connections':
        model = models.SimpleUNet3D(activation_func = config.activation_function, 
                                    in_channels     = 1, 
                                    out_channels    = 1, 
                                    exp_type        = config.experiment_type)
        return model
    elif which_model == 'unet_skip_connections':
        model = models.UNet3D(activation_func = config.activation_function,
                              in_channels     = 1,
                              out_channels    = 1,
                              exp_type        = config.experiment_type)
        return model
    elif which_model == 'combnet_v1':
        # encoder graph-unet and decoder without skip connections
        model = models.CombNet_v1(activation_func_unet  = config.activation_function,
                                  activation_func_graph = config.activation_function_g,
                                  in_channels_unet      = 1,
                                  hidden_channels_graph = config.hidden_channels_g,
                                  depth_graph           = config.depth_g,
                                  pool_ratios_graph     = config.pool_ration_g,
                                  sum_res_graph         = config.sum_res_g,
                                  out_channels_unet     = 1,
                                  exp_type              = config.experiment_type
        )
        return model
    elif which_model == 'combnet_v2':
        model = models.CombNet_v2()
        return model 
    else:
        raise NotImplementedError(f'The model {which_model} is not implemented')

def get_optimizer(model, which_optimizer, lr=0.01, wd=0.01):
    '''
    This function initializes the optimizer

    Parameters
    ----------
    model: The used model
    which_optimizer: The options are:
        1) adam
        2) ...
    lr: learning rate
    wd: weight decay
    Returns
    -------
    optimizer
    '''
    if which_optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = wd)
        return optimizer
    elif which_optimizer.lower() == 'adamw':
        # Good for image classification. To achieve state-of-the-art results is used 
        # with OneCycleLR scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = wd)
        return optimizer
    elif which_optimizer.lower() == 'adagrad':
        # You can avoid using a scheduler because, Adagrad dynamically adapts the learning
        # rate itself. It is used for sparse datasets.
        optimizer = torch.optim.Adagrad(model.parameters(), lr = lr, weight_decay = wd)
        return optimizer

def get_loss(which_loss):
    '''
    This function defines the criterion used for optimizing

    Parameters
    ----------
    which_loss: The options are:
        1) ...
        2) ...

    Returns
    -------
    criterion
    '''
    if which_loss == 'dice_loss':
        return dice_loss
    elif which_loss ==  'mse_dice_loss':
        return mse_dice_loss
    return None

#---------- losses
def dice_loss(batch_preds, batch_targets, smooth = 1e-05, reduction = 'mean'):
    
    pflat        = batch_preds.float().contiguous().view(batch_preds.shape[0], -1)
    tflat        = batch_targets.float().contiguous().view(batch_targets.shape[0], -1)
    intersection = torch.sum(torch.mul(pflat, tflat), dim = 1)
    nom          = 2. * intersection + smooth
    denom        = torch.sum(pflat, dim=1) + torch.sum(tflat, dim = 1) + smooth
    dice_losses  = 1 - nom/denom
    
    if reduction == 'mean':
        loss = torch.mean(dice_losses)
    elif reduction == 'sum':
        loss = torch.sum(dice_losses)
    else:
        raise NotImplementedError(f'Dice loss reduction {reduction} not implemented') 
    return loss 

def mse_dice_loss(node_fts_preds, node_fts_gt, adj_preds, adj_gt, adj_mtx):
    # TO BE IMPLEMENTED
    d_loss = dice_loss(node_fts_preds, node_fts_gt)
    print((adj_mtx-adj_preds).any() != 0)
    return d_loss

def dice_score_metric(batch_preds, batch_targets, smooth = 1e-05):
    
    pflat        = batch_preds.float().contiguous().view(batch_preds.shape[0], -1)
    tflat        = batch_targets.float().contiguous().view(batch_targets.shape[0], -1)
    intersection = torch.sum(torch.mul(pflat, tflat), dim = 1)
    nom          = 2. * intersection + smooth
    denom        = torch.sum(pflat, dim=1) + torch.sum(tflat, dim = 1) + smooth

    dice_scores      = nom/denom
    dice_scores_list = dice_scores.view(-1).cpu().tolist()
    
    del dice_scores
    return dice_scores_list

