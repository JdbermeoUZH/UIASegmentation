'''
As the name suggest this script split the dataset in training and test set.
The split occured with the constrained to keep the distribution of aneurysm tags
the same in both training and test set. Because the training data are few, we
use kfold split in order further divide the training dataset in training and
validation set.
'''

import re
import os
import time
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from preprocessing_utils import get_subjects_folders, find_types_of_aneurysm

## paths
save_path       = "/scratch/kvergopoulos/SemesterProject/datasets"
path_to_dataset = "/scratch/kvergopoulos/SemesterProject/datasets/USZ_BrainArtery_Originals"

# possible aneurysm classes, given by Elisa
aneurysm_classes = ['mca', 'acomm', 'ica', 'pcomm', 'pica', 'aca', 
                    'pca', 'ba', 'va', 'multiple', 'not_classified']

def split_dataset(path_to_dataset, save_path, aneurysm_classes):
    '''
    This function split the dataset on training validation and testing dataset
    maintaining the initial distribution of the data labels.

    Parameters
    ----------
    save_path: save destination for the file containing the splits
    path_to_dataset: path to original dataset
    aneurysm_classes: all possible aneurysm classes 

    Returns
    -------
    None
    '''
    
    paths_list  = get_subjects_folders(path_to_dataset)
    random.seed(time.time())
    random.shuffle(paths_list)

    names_list  = [i.split('/')[-1] for i in paths_list]
    classes_pop = defaultdict(int) 
    classes     = list()

    for t_name in names_list:
        # split tags
        name_components = re.split(r'_|-', t_name.lower())
        # find which aneurysm types exist
        types_of_aneur = find_types_of_aneurysm(aneurysm_classes, name_components)

        if len(types_of_aneur) == 0:
            print(f"Subject {t_name} with none aneurysm type")
            # append it as not classified???
            classes.append('not_classified')
            classes_pop['not_classified'] += 1

        elif len(types_of_aneur) == 1 and types_of_aneur[0]!= 'not_classified':
            classes.append(types_of_aneur[0])
            classes_pop[types_of_aneur[0]] += 1
        
        elif len(types_of_aneur)>1:
            classes.append('multiple')
            classes_pop['multiple'] += 1

        else:
            classes.append('not_classified')
            classes_pop['not_classified'] += 1
    assert len(classes) == len(names_list)

    # split dataset to train & test set keeping the distribution
    train_data, test_data, train_classes, test_classes = \
        train_test_split(names_list, classes, test_size=int(len(names_list)*0.2))
    
    # creat k-folds with k = 5
    k_fold_dict = dict()
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for i, (train_indexes, valid_indexes) in enumerate(skf.split(train_data, train_classes)):
        k_fold_dict[i] = {'train_data': [train_data[i] for i in train_indexes],
                          'valid_data': [train_data[i] for i in valid_indexes],
                          'test_data':  test_data
                          }    
    pd.DataFrame(k_fold_dict).to_json(os.path.join(save_path, 'k_fold_val5.json'))

###
split_dataset(path_to_dataset, save_path, aneurysm_classes)