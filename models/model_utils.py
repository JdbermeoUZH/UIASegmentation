import os
import csv
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim as optim
from models import uia_models as models
from torch_geometric.utils import coalesce

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
                                    out_channels    = config.output_channels, 
                                    exp_type        = config.experiment_type)
        return model
    elif which_model == 'unet_skip_connections':
        model = models.UNet3D(activation_func = config.activation_function,
                              in_channels     = 1,
                              out_channels    = config.output_channels,
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
                                  out_channels_unet     = config.output_channels,
                                  exp_type              = config.experiment_type
                                  )
        return model
    elif which_model == 'combnet_v2':
        model = models.CombNet_v2(activation_func_unet  = config.activation_function,
                                  activation_func_graph = config.activation_function_g,
                                  in_channels_unet      = 1,
                                  hidden_channels_graph = config.hidden_channels_g,
                                  depth_graph           = config.depth_g,
                                  pool_ratios_graph     = config.pool_ration_g,
                                  sum_res_graph         = config.sum_res_g,
                                  out_channels_unet     = config.output_channels,
                                  exp_type              = config.experiment_type
                                  )
        return model 
    elif which_model == 'unet_baseline':
        model = models.UNet3D(activation_func = config.activation_function,
                              in_channels     = 1,
                              out_channels    = config.output_channels,
                              exp_type        = config.experiment_type)
        return model
    elif which_model == 'combnet_v3':
        model = models.CombNet_v3(activation_func_unet  = config.activation_function,
                                  activation_func_graph = config.activation_function_g,
                                  in_channels_unet      = 1,
                                  hidden_channels_graph = config.hidden_channels_g,
                                  depth_graph           = config.depth_g,
                                  out_channels_unet     = config.output_channels,
                                  exp_type              = config.experiment_type
                                  )
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
    elif which_loss ==  'graph_loss':
        return graph_loss
    return None

def get_evaluation_metrics():
    metrics = {
        'dice_score':      DiceScore,
        'dice_score_soft': DiceScoreSoft,
        'recall':          Recall,
        'precission':      Precision
    }
    return metrics

#---------- LOSSES
def dice_loss(batch_preds, batch_targets, smooth = 1e-05, reduction = 'mean'):
    
    # TODO: check if .contiguous is unnecesary.
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

def graph_loss(node_fts_preds, node_fts_gt, adj_mtx_preds, adj_mtx_weight, adj_mtx_gt):
    
    # compute adjacency matrix loss using mean square error in sparse matrices
    adj_loss = []
    for image in range(adj_mtx_preds.shape[0]):
        indices      = torch.cat((adj_mtx_preds[image], adj_mtx_gt[image]), dim = 1)
        init_weights = -1 * node_fts_preds.new_ones(adj_mtx_gt[image].size(1)) 
        vals         = torch.cat((adj_mtx_weight[image], init_weights))
        inds, values = coalesce(indices, vals, num_nodes=node_fts_gt.shape[1])
        adj_loss.append(torch.mean(values))
    adj_loss = torch.stack(adj_loss)
    
    # compute node features loss
    d_loss = dice_loss(node_fts_preds, node_fts_gt)
    return d_loss + torch.mean(adj_loss)

#---------- HELPER CLASSES
class LRScheduler():
    def __init__(self, optimizer, train_dataloader, nepochs, which_scheduler, config):
        
        self.sch            = None
        self.optimizer      = optimizer
        self.scheduler_name = which_scheduler.lower()

        if which_scheduler.lower() == 'reduce_lr_on_plateau':
            self.sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, 
                                                            mode      = 'min', 
                                                            factor    = 0.1, 
                                                            patience  = int(config.patience*0.2),
                                                            min_lr    = 1e-6,
                                                            verbose   = True)
        elif which_scheduler.lower() == 'one_cylce_lr':
            self.sch = optim.lr_scheduler.OneCycleLR(optimizer       = self.optimizer,
                                                     max_lr          = 0.01,
                                                     steps_per_epoch = len(train_dataloader),
                                                     epochs          = nepochs,
                                                     verbose         = True)
        elif which_scheduler.lower() == 'step_lr':
            self.sch = optim.lr_scheduler.StepLR(optimizer  = self.optimizer, 
                                                 step_size  = int(nepochs*0.1), 
                                                 gamma      = 0.005, 
                                                 last_epoch = -1,
                                                 verbose    = True)
    def __call__(self, val_loss):
        if self.sch == None: pass
        elif self.scheduler_name == 'step_lr': self.sch.step()
        else: self.sch.step(val_loss)

class EarlyStopping():
    def __init__(self, patience = 5, min_delta = 0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_epoch = -1
        self.best_loss  = None
        self.early_stop = False
    
    def __call__(self, val_loss, tepoch):
        if self.best_loss == None:
            self.best_loss  = val_loss
            self.best_epoch = tepoch
            self.counter    = 0
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = tepoch
            self.counter    = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
            if self.counter > self.patience:
                print(f'INFO: Early stopping encountered. Best epoch {self.best_epoch}')
                self.early_stop = True
        return self.early_stop

class MSaver():
    def __init__(self, path_to_save, experiment_name, update_rate = 20, check = True):
        
        if os.path.exists(path_to_save) == False:   os.makedirs(path_to_save)
        self.path_to_save = path_to_save + '/' + experiment_name
        if os.path.exists(self.path_to_save) == False: os.makedirs(self.path_to_save)

        self.experiment_name = experiment_name
        self.update_rate     = update_rate
        self.counter         = 0
        self.check           = check
        if self.check == True:  self.current_best = None 
        
    def __call__(self, model, dice_score):
        
        self.counter += 1
        if self.counter > self.update_rate:
            self.counter = 0
            self.save_model(model, self.path_to_save + '/' + self.experiment_name + '_r.mod')
        
        if self.check == True:
            if self.current_best == None:
                self.current_best = dice_score
                self.save_model(model, self.path_to_save + '/' + self.experiment_name + '_b.mod')
            elif self.current_best < dice_score:
                self.current_best = dice_score
                self.save_model(model, self.path_to_save + '/' + self.experiment_name + '_b.mod')

    def save_model(self, model, name_path=''):
        if name_path == '':
            name_path = self.path_to_save + '/' + self.experiment_name + '_e.mod'
        torch.save(model.state_dict(), name_path)


#---------- EVALUATION METRICS
class MetricsCollector():
    def __init__(self, eval_metrics, config, testing_path = ''):
        self.log_dict = dict()
        self.config   = config
        self.metrics  = eval_metrics
        if testing_path == '':
            if os.path.exists(config.path_results) == False:
                os.makedirs(config.path_results)
            
            self.path     = self.config.path_results + '/' + self.config.experiment_name
            if os.path.exists(self.path) == False:
                os.makedirs(self.path)
        else:
            self.path = testing_path
            if os.path.exists(self.path) == False:
                os.makedirs(self.path)
    
    def save_config(self):
        self.file_path = self.path + '/' + 'config.txt'
        temp_dict      = vars(self.config)
        with open(self.file_path, "w") as file: 
            for key, value in temp_dict.items():
                line = f'{key}: {value} \n'
                file.write(line)
        del temp_dict

    def add_epoch(self, eval_epoch, tepoch, loss_train, loss_val):
        
        if self.log_dict.get(tepoch) is not None:
            raise KeyError(f'In Metrics Collector: the epoch {tepoch} has already been documented')
        
        eval_dict  = eval_epoch.get_average_metrics()
        epoch_dict = {'train_loss': loss_train,
                      'valid_loss': loss_val}
        epoch_dict.update(eval_dict)
        
        self.log_dict[tepoch] = epoch_dict
    
    def save_logs(self):
        self.logs_path = self.path + '/' + 'logs.txt'
        header         = ['epochs']
        keys           = list(self.log_dict.keys())
        header        += list(self.log_dict[keys[0]].keys())
        
        with open(self.logs_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for key, epoch_dict in self.log_dict.items():
                line = [key]
                for _, value in epoch_dict.items():  line.append(value)
                writer.writerow(line)

    def save_training_figs(self):
        # plot training and validation loss
        epochs   = list()
        tr_loss  = list()
        val_loss = list()
        for k_epoch, epoch_dict in self.log_dict.items():
            epochs.append(k_epoch)
            tr_loss.append(epoch_dict['train_loss'])
            val_loss.append(epoch_dict['valid_loss'])
        
        self.losses_fig_path = self.path + '/' + 'train_valid_losses.png'
        plt.plot(epochs, tr_loss, label = 'training_loss')
        plt.plot(epochs, val_loss, label = 'validation_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(self.losses_fig_path)

    def print_best_metrics(self):
        for name in self.metrics.keys():
            b_epoch      = -1
            for epoch, epoch_dict in self.log_dict.items():
                if b_epoch == -1:
                    b_epoch = epoch
                else:
                    if epoch_dict[name] > self.log_dict[b_epoch][name]:
                        b_epoch = epoch
            print(f'for metric {name} the best epoch was {b_epoch}')
            print(self.log_dict[b_epoch])
            
class MetricsClass():
    def __init__(self, metrics):
        
        if isinstance(metrics, dict):
            self.metrics  = metrics
        else:
            raise TypeError(f'In Metrics Class: the metrics variable is of type {type(metrics)}')
        
        self.avg_results = None
        self.results     = dict()
        for key in self.metrics.keys(): self.results[key] = []
        
    def __call__(self, node_preds, node_targets):
        '''
        the node matrices should have shapes:
        (number of images, number of batches, number of channels, size_x, size_y, size_z)
        '''

        for image in range(node_preds.shape[0]):
            preds_bin  = binarize_image(node_preds[image])
            target_bin = binarize_image(node_targets[image]) 
            for key, func in self.metrics.items():
                try:
                    if key == 'dice_score_soft': 
                        res = func(node_preds[image], node_targets[image])
                    else:   
                        res = func(preds_bin, target_bin)
                except Exception as e:
                    print(f'Error {e} in the calculation of metric {key}')
                    res = np.NAN

                if isinstance(res, torch.Tensor):   res = res.cpu().item()
                self.results[key].append(res)
            del preds_bin, target_bin
    
    def compute_average_metrics(self):
        self.avg_results = dict()
        for name, metric_results in self.results.items():
            self.avg_results[name] = np.nanmean(metric_results)

    def get_average_metrics(self):
        self.compute_average_metrics()
        return self.avg_results
    
    def get_metric(self, metric_name):
        if self.results.get(metric_name.lower()) is None:
            raise ValueError(f'In Metric class: the {metric_name.lower()} has not been calculated')
        res = np.nanmean(self.results[metric_name])
        return res
    
    def print_aggregate_results(self):
        self.compute_average_metrics()
        msg = ''
        for key, val in self.avg_results.items():
            msg += f'{key}: {val:.5f}, '
        msg += '\n'
        print(msg)

def binarize_image(img, threshold = 0.5, one_hot = False):
    if img.ndim == 4:
       img = img.unsqueeze(0)
    
    assert img.ndim == 5, f'Binarize_image, tensor mismatch {img.shape}'

    n_channels = img.shape[1]

    # binary problem
    if n_channels == 1:
        nimg = img > threshold
    else:
        print("FUCKKK")

    nimg = nimg.float()
    return nimg

def DiceScore(image_preds, image_target):

    pflat        = image_preds.view(-1)
    tflat        = image_target.view(-1)

    tp           = torch.sum((pflat == 1) & (tflat == 1)) 
    fn           = torch.sum((pflat == 0) & (tflat == 1))
    fp           = torch.sum((pflat == 1) & (tflat == 0))
    if 2*tp + fp + fn == 0: return np.NaN
    return 2*tp/(2*tp + fp + fn)

def DiceScoreSoft(image_preds, image_target, smooth = 1e-05):
    
    pflat        = image_preds.float().view(-1)
    tflat        = image_target.float().view(-1)
    intersection = torch.sum(torch.mul(pflat, tflat), dim = 0)
    nom          = 2. * intersection + smooth
    denom        = torch.sum(pflat) + torch.sum(tflat) + smooth
    dice_soft    = nom/denom
    return dice_soft.cpu().item()
 
def Recall(image_preds, image_target):
    pflat = image_preds.view(-1)
    tflat = image_target.view(-1)

    tp = torch.sum((pflat == 1) & (tflat == 1))
    fn = torch.sum((pflat == 0) & (tflat == 1))

    if tp + fn == 0: return np.NaN
    return tp/(tp+fn)

def Precision(image_preds, image_target):
    pflat = image_preds.view(-1)
    tflat = image_target.view(-1)
    tp = torch.sum((pflat == 1) & (tflat == 1))
    fp = torch.sum((pflat == 1) & (tflat == 0))
    if fp + tp == 0: return 1e-12
    return tp/(tp+fp)

# to remove
def dice_score_metric(batch_preds, batch_targets, smooth = 1e-05):
    
    pflat        = batch_preds.float().contiguous().view(batch_preds.shape[0], -1)
    tflat        = batch_targets.float().contiguous().view(batch_targets.shape[0], -1)
    intersection = torch.sum(torch.mul(pflat, tflat), dim = 1)
    nom          = 2. * intersection + smooth
    denom        = torch.sum(pflat, dim=1) + torch.sum(tflat, dim = 1) + smooth

    dice_scores      = nom/denom
    dice_scores_list = dice_scores.view(-1).cpu().tolist()
    
    del dice_scores, pflat, tflat, intersection, nom, denom
    return dice_scores_list

#---------- TESTING FUNCTIONS
def model_predict_single(model_path, test_dataloader, config, path_to_save_results):
    
    model = load_model(model_path, config)

    #---------- logs and metrics
    eval_metrics   = get_evaluation_metrics() 
    test_collector = MetricsCollector(eval_metrics, config, path_to_save_results)

    #---------- Testing LOOP
    model.eval()
    test_epoch       = MetricsClass(eval_metrics)
    # each batch contains n_images, n_patches, 1 channel, patch_size_x, patch_size_y, patch_size_z
    with tqdm(test_dataloader, unit='batch') as tqdm_loader:
        for adj_mtx, fts, adj_mtx_gt, fts_gt in tqdm_loader:
            #---------- test model
            if config.use_patches == True:
                if config.only_unets_flag == True:
                    #print("INFO: test-v1")
                    fts_preds = test_v1(model, fts, fts_gt, config)
                elif config.only_unets_flag == False:
                    #print("INFO: test-v2")
                    fts_preds = test_v2(model, adj_mtx, fts, adj_mtx_gt, fts_gt, config)
            elif config.use_patches == False:
                #print("INFO: test-v3")
                fts_preds = test_v3(model, fts, fts_gt, config)
            
            test_epoch(fts_preds, fts_gt)
    test_collector.add_epoch(test_epoch, 0, 0, 0)


def test_v1(model, node_fts, node_fts_gt, config):
    
    with torch.no_grad():    
        node_fts       = node_fts.to(config.device)
        node_fts_gt    = node_fts_gt.to(config.device)
        batch_shape    = node_fts.shape
                    
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
    return batch_preds

def test_v2(model, adj_mtx, node_fts, adj_mtx_gt, node_fts_gt, config):
    with torch.no_grad():
        node_fts       = node_fts.to(config.device)
        adj_mtx        = adj_mtx.to(config.device)

        node_preds, adj_preds = model(node_fts, adj_mtx)
    return node_preds

def test_v3(model, image, segm_image, config):
    with torch.no_grad():
        
        image      = image.to(config.device)
        pred       = model(image)
    return pred

def load_model(model_path, config):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model path {model_path} does not exists')
    
    model = get_model(config.which_net, config)
    model.load_state_dict(model_path)
    model.to(config.device)
    model.eval()
    return model
