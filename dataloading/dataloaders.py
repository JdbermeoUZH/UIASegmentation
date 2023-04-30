def get_dataloaders_all(config, split_dict):

    train_loader = get_dataloader_single(split_dict['train'],
                                         config.batch_size,
                                         config.num_workers,
                                         )
    valid_loader = get_dataloader_single(split_dict['valid'],
                                         config.batch_size_val,
                                         config.num_workers)
    
    test_loader  = get_dataloader_single(split_dict['test'],
                                         config.batch_size_test,
                                         config.num_workers
                                         )

    return train_loader, valid_loader, test_loader

def get_dataloader_single():
    return None