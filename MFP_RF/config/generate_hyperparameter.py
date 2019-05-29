from __future__ import print_function

import json
import random
from sklearn.model_selection import ParameterGrid


def generate_hyperparameter_for_random_forest():
    upper_limit = 1000
    conf = {
        'enrichment_factor': {
            'ratio_list': [0.02, 0.01, 0.0015, 0.001]
        },
        'random_seed': 1337
    }

    hyperparameter_sets = {
        'n_estimators': [50, 250, 1000, 4000, 8000, 16000],
        'max_features': [None, 'sqrt', 'log2'],
        'min_samples_leaf': [1, 10, 100, 1000],
        'class_weight': [None, 'balanced_subsample','balanced']
    }

    hyperparameters = ParameterGrid(hyperparameter_sets)
    hyperparameters_list = map(lambda x: x, hyperparameters)

    random.seed(1337)
    random.shuffle(hyperparameters_list)

    for cnt, param in enumerate(hyperparameters_list):
        conf['n_estimators'] = param['n_estimators']
        conf['max_features'] = param['max_features']
        conf['min_samples_leaf'] = param['min_samples_leaf']
        conf['class_weight'] = param['class_weight']

        with open('random_forest_classification/{}.json'.format(cnt), 'w') as file_:
            json.dump(conf, file_)

        if cnt >= upper_limit:
            break


if __name__ == '__main__':
    generate_hyperparameter_for_random_forest()
