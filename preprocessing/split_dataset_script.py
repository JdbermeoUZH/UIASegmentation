'''
Description of the script 
For now I will hardcode the path to data. Maybe change it later
'''

import re
import os
import json
import time
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from preprocessing_utils import get_subjects_folders, find_types_of_aneurysm

## paths
save_path       = "/scratch/kvergopoulos/SemesterProject/datasets"
path_to_dataset = "/scratch/kvergopoulos/SemesterProject/datasets/USZ_BrainArtery_Originals"


# possible aneurysm classes, given by Elisa
aneurysm_classes = ['mca', 'acomm', 'ica', 'pcomm', 'pica', 'aca', 
                    'pca', 'ba', 'va', 'multiple', 'not_classified']

def split_dataset(path_to_dataset, save_path, aneurysm_classes):
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

    # split dataset to train & test set
    test_indexes = random.sample([i for i in range(len(names_list))], 
                                int(0.2 * len(names_list)))
    train_indexes = [i for i in range(len(names_list)) if i not in test_indexes]
    train_data    = [names_list[i] for i in train_indexes]
    train_classes = [classes[i] for i in train_indexes]
    test_data     = [names_list[i] for i in test_indexes]
    test_classes  = [classes[i] for i in test_indexes]

    # creat k-folds with k = 5
    k_fold_dict = dict()
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for i, (train_indexes, valid_indexes) in enumerate(skf.split(train_data, train_classes)):

        k_fold_dict[i] = {'train_data': [train_data[i] for i in train_indexes],
                          'valid_data': [train_data[i] for i in valid_indexes],
                          'test_data':  test_data
                          }
    
    # test
    for i in k_fold_dict:
        print(len(k_fold_dict[i]['train_data']), len(k_fold_dict[i]['valid_data']), len(k_fold_dict[i]['test_data']))
    pd.DataFrame(k_fold_dict).to_json(os.path.join(save_path, 'k_fold_val5.json'))
split_dataset(path_to_dataset, save_path, aneurysm_classes)