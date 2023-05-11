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
        model = models.SimpleUNet3D(config.activation_function, 1, 1)
        return model

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
        # with OneCycleLR
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
    return None

#---------- losses
def dice_loss(batch_preds, batch_targets, smooth = 1e-05, sigmoid = True, reduction = 'mean'):
    print("mpeeneiiiii edw???")
    print(batch_preds.shape)
    print(batch_targets.shape)

    batch_preds   = batch_preds.float()
    batch_targets = batch_targets.float()

    if sigmoid == True:
        batch_preds = torch.sigmoid(batch_preds)
    
    dice_losses = []
    for image in range(batch_preds.shape[0]):
        pflat        = batch_preds[image, :, :, :, :, :].contiguous().view(-1)
        tflat        = batch_targets[image, :, :, :, :, :].contiguous().view(-1)
        intersection = torch.sum(pflat * tflat)
        print(intersection)
        print(torch.sum(pflat), torch.sum(tflat))
        dice_loss    = 1 - ((2. * intersection + smooth)/(torch.sum(pflat) + torch.sum(tflat) + smooth))
        dice_losses.append(dice_loss)
    dice_losses = torch.stack(dice_losses)
    
    if reduction == 'mean':
        loss = torch.mean(dice_losses)
    elif reduction == 'sum':
        loss = torch.sum(dice_losses)
    else:
        raise NotImplementedError(f'Dice loss reduction {reduction} not implemented')
    print(loss)
    return loss