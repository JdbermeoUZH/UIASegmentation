import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from models import model_utils as mu
from dataloading import test_datasets
from torch.utils.data import DataLoader
from dataloading import dataloaders as dl

#---------- TESTING FUNCTIONS
def save_images(name, pred_image, segm_image, model_dict):
    
    pred_image_bin  = mu.binarize_image(pred_image).detach().cpu().numpy()
    pred_path       = model_dict['images_path'] + '/' + name + '_pred.npy'
    segm_image_bin  = mu.binarize_image(segm_image).detach().cpu().numpy()
    segm_path       = model_dict['images_path'] + '/' + name +  '_segm.npy'
    np.save(pred_path, pred_image_bin)
    np.save(segm_path, segm_image_bin)


def save_individual_metrics(scores_dict, path):
    header  = ['names']
    keys    = list(scores_dict.keys())
    header += list(scores_dict[keys[0]].keys())
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for key, stats_dict in scores_dict.items():
            line = [key]
            for _, value in stats_dict.items(): line.append(value)
            writer.writerow(line)


def get_testdataloader(type_of_loader, 
                       data, 
                       path_data, 
                       batch_size, 
                       shuffle,
                       num_workers,
                       config):
    
    if config.use_patches == True:
        transform = dl.get_transform_test(config)
        dataset   = test_datasets.UIAGraph_Dataset(path_data, data, transform, config)
    elif config.use_patches == False:
        transform = dl.get_transform_test(config)
        dataset   = test_datasets.UIA_Dataset(path_data, data, transform, config)
    
    custom_loader = DataLoader(dataset,
                               batch_size  = batch_size,
                               shuffle     = shuffle,
                               num_workers = num_workers,
                               pin_memory  = False)
    return custom_loader
    

def load_model_weights(model_path, config, device):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model path {model_path} does not exists')
    
    model = mu.get_model(config.which_net, config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)
    return model


def model_predict_single(test_dataloader, config, model_dict):
    
    #---can also run locally
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = load_model_weights(model_dict['model_path'], config, device)

    #---------- logs and metrics
    eval_metrics   = mu.get_evaluation_metrics() 
    test_collector = mu.MetricsCollector(eval_metrics, config, model_dict['path_to_save_test'])
    test_epoch     = mu.MetricsClass(eval_metrics)
    if model_dict['save_extend']:   individual_scores = dict()

    #---------- Testing LOOP
    if config.which_net == 'combnet_v4':
        print("INFO: Testing-v4 started")
        with tqdm(test_dataloader, unit='batch') as tqdm_loader:
            for name, adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                name = name[0]
                print(f'predict for {name}')
                
                node_fts_preds = test_model_v4(model, node_fts, adj_mtx, node_fts_gt, device)
                test_epoch(node_fts_preds, node_fts_gt)
                
                if model_dict['save_extend']:
                    individual_scores[name] = test_epoch.get_last_item()
                    # TODO: reshape the patches back to one piece and save the image
                    #save_images(name, pred_image, segm_image, model_dict)

    elif config.which_net == 'unet_baseline':
        print("INFO: Testing-v3 started")
        with tqdm(test_dataloader, unit='batch') as tqdm_loader:
            for name, _, image, _, segm_image in tqdm_loader:
                name = name[0]
                print(f'predict for {name}')
                
                pred_image = test_model_v3(model, image, segm_image, device)
                test_epoch(pred_image, segm_image)
                
                if model_dict['save_extend']:
                    individual_scores[name] = test_epoch.get_last_item()
                    save_images(name, pred_image, segm_image, model_dict)

    elif config.which_net == 'combnet_v5':
        print("INFO: Testing-v5 started")
        with tqdm(test_dataloader, unit='batch') as tqdm_loader:
            for name, adj_mtx, image, adj_mtx_gt, segm_image in tqdm_loader:
                name = name[0]
                print(f'predict for {name}')

                pred_image = test_model_v5(model, image, adj_mtx, segm_image, device)
                test_epoch(pred_image, segm_image)

                if model_dict['save_extend']:
                    individual_scores[name] = test_epoch.get_last_item()
                    save_images(name, pred_image, segm_image, model_dict)
    
    #---------- save results and metrics
    test_epoch.print_aggregate_results()
    test_collector.add_epoch(test_epoch, 0, 0, 0)
    test_collector.save_config()
    test_collector.save_logs()
    
    if model_dict['save_extend']:
        save_individual_metrics(individual_scores, model_dict['ind_scores_path'])    
    return 


def ensemble_model_predict(test_dataloader, config, model_dict):
    
    #---can also run locally
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = list()
    
    for model_path in model_dict['models']:
        try:
            model = load_model_weights(model_path, config, device)
        except Exception as e:
            print(f'ERROR: Could not load model from path: {model_path}... Exiting')
            print(e)
            raise RuntimeError
        models.append(model)
    print("INFO: Models loading completed")

    #---------- logs and metrics
    eval_metrics   = mu.get_evaluation_metrics() 
    test_collector = mu.MetricsCollector(eval_metrics, config, model_dict['path_to_save_test'])
    test_epoch     = mu.MetricsClass(eval_metrics)
    if model_dict['save_extend']:   individual_scores = dict()

    #---------- Testing LOOP
    if config.which_net == 'combnet_v4':
        print("INFO: Testing-v4 started")
        with tqdm(test_dataloader, unit='batch') as tqdm_loader:
            for name, adj_mtx, node_fts, adj_mtx_gt, node_fts_gt in tqdm_loader:
                name = name[0]
                print(f'predict for {name}')
                
                node_fts_preds = None
                for model in models:
                    node_fts_temp = test_model_v4(model, node_fts, adj_mtx, node_fts_gt, device)
                    if node_fts_preds == None:
                        if model_dict['en_method'] == 'voting': node_fts_preds = mu.binarize_image(node_fts_temp)
                        elif model_dict['en_method'] == 'mean_aggr': node_fts_preds = node_fts_temp
                    else:
                        if model_dict['en_method'] == 'voting': node_fts_preds += mu.binarize_image(node_fts_temp)
                        elif model_dict['en_method'] == 'mean_aggr': node_fts_preds += node_fts_temp
                if model_dict['en_method'] == 'voting': node_fts_preds = torch.where(node_fts_preds >= len(models)/2.0, 1, 0)
                elif model_dict['en_method'] == 'mean_aggr': node_fts_preds = node_fts_preds/len(models)
                
                test_epoch(node_fts_preds, node_fts_gt)
                if model_dict['save_extend']:
                    individual_scores[name] = test_epoch.get_last_item()
                    # TODO: reshape the patches back to one piece and save the image
                    #save_images(name, pred_image, segm_image, model_dict)

    elif config.which_net == 'unet_baseline':
        print("INFO: Testing-v3 started")
        with tqdm(test_dataloader, unit='batch') as tqdm_loader:
            for name, _, image, _, segm_image in tqdm_loader:
                name = name[0]
                print(f'predict for {name}')

                pred_image = None
                for model in models:
                    pred_temp = test_model_v3(model, image, segm_image, device)
                    if pred_image == None:
                        if model_dict['en_method'] == 'voting': pred_image = mu.binarize_image(pred_temp)
                        elif model_dict['en_method'] == 'mean_aggr': pred_image = pred_temp
                    else:
                        if model_dict['en_method'] == 'voting': pred_image += mu.binarize_image(pred_temp)
                        elif model_dict['en_method'] == 'mean_aggr': pred_image += pred_temp
                if model_dict['en_method'] == 'voting': pred_image = torch.where(pred_image >= len(models)/2.0, 1, 0)
                elif model_dict['en_method'] == 'mean_aggr': pred_image = pred_image/len(models)
                
                test_epoch(pred_image, segm_image)
                if model_dict['save_extend']:
                    individual_scores[name] = test_epoch.get_last_item()
                    #save_images(name, pred_image, segm_image, model_dict)

    elif config.which_net == 'combnet_v5':
        print("INFO: Testing-v5 started")
        with tqdm(test_dataloader, unit='batch') as tqdm_loader:
            for name, adj_mtx, image, adj_mtx_gt, segm_image in tqdm_loader:
                name = name[0]
                print(f'predict for {name}')

                pred_image = None
                for model in models:
                    pred_temp = test_model_v5(model, image, adj_mtx, segm_image, device)
                    if pred_image == None:
                        if model_dict['en_method'] == 'voting': pred_image = mu.binarize(pred_temp)
                        elif model_dict['en_method'] == 'mean_aggr': pred_image = pred_temp
                    else:
                        if model_dict['en_method'] == 'voting': pred_image += mu.binarize_image(pred_temp)
                        elif model_dict['en_method'] == 'mean_aggr': pred_image += pred_temp
                if model_dict['en_method'] == 'voting': pred_image = torch.where(pred_image >= len(models)/2.0, 1, 0)
                elif model_dict['en_method'] == 'mean_aggr': pred_image = pred_image/len(models)

                test_epoch(pred_image, segm_image)
                if model_dict['save_extend']:
                    individual_scores[name] = test_epoch.get_last_item()
                    #save_images(name, pred_image, segm_image, model_dict)
    
    #---------- save results and metrics
    test_epoch.print_aggregate_results()
    test_collector.add_epoch(test_epoch, 0, 0, 0)
    test_collector.save_config()
    test_collector.save_logs()
    
    if model_dict['save_extend']:
        save_individual_metrics(individual_scores, model_dict['ind_scores_path'])    
    return

def test_model_v3(model, image, segm_image, device):
    with torch.no_grad():
        image      = image.to(device)
        segm_image = segm_image.to(device)
        pred       = model(image)
    return pred

def test_model_v4(model, node_fts, adj_mtx, node_fts_gt, device):
    with torch.no_grad():
        node_fts    = node_fts.to(device)
        adj_mtx     = adj_mtx.to(device)
        node_fts_gt = node_fts_gt.to(device)
        node_fts_preds, _, _ = model(node_fts, adj_mtx)
    return node_fts_preds

def test_model_v5(model, image, adj_mtx, segm_image, device):
    with torch.no_grad():
        image      = image.to(device)
        adj_mtx    = adj_mtx.to(device)
        segm_image = segm_image.to(device)
        pred_image, _, _ = model(image, adj_mtx)
    return pred_image
