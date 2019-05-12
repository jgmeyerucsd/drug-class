from __future__ import print_function

from data_loader import *
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from util import output_classification_result, reshape_data_into_2_dim


class RandomForestClassification:
    def __init__(self, conf):
        self.conf = conf
        self.max_features = conf['max_features']
        self.n_estimators = conf['n_estimators']
        self.min_samples_leaf = conf['min_samples_leaf']
        self.class_weight = conf['class_weight']
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        self.random_seed = conf['random_seed']

        if 'hit_ratio' in self.conf.keys():
            self.hit_ratio = conf['hit_ratio']
        else:
            self.hit_ratio = 0.01
        np.random.seed(seed=self.random_seed)
        return

    def setup_model(self):
        model = RandomForestClassifier(n_estimators=self.n_estimators,
                                       max_features=self.max_features,
                                       min_samples_leaf=self.min_samples_leaf,
                                       n_jobs=8,
                                       class_weight=self.class_weight,
                                       random_state=self.random_seed,
                                       oob_score=False,
                                       verbose=1)
        return model

    def train_and_predict(self, x_train, y_train, x_test, y_test, weight_file):
        model = self.setup_model()
        model.fit(x_train, y_train)

        y_pred_on_train = reshape_data_into_2_dim(model.predict(x_train))
        if x_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict(x_test))
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test)
        np.savez('output_aliper/num_class_{}_index_{}'.format(number_of_class, index),
                 y_train=y_train, y_pred_on_train=y_pred_on_train,
                 y_test=y_test, y_pred_on_test=y_pred_on_test)

        self.save_model(model, weight_file)

        return

    def predict_with_existing(self, X_data, weight_file):
        model = self.load_model(weight_file)
        y_pred = reshape_data_into_2_dim(model.predict_proba(X_data)[:, 1])
        return y_pred

    def eval_with_existing(self, x_train, y_train, x_test, y_test, weight_file):
        model = self.load_model(weight_file)

        y_pred_on_train = reshape_data_into_2_dim(model.predict(x_train))
        if x_test is not None:
            y_pred_on_test = reshape_data_into_2_dim(model.predict(x_test))
        else:
            y_pred_on_test = None

        output_classification_result(y_train=y_train, y_pred_on_train=y_pred_on_train,
                                     y_val=None, y_pred_on_val=None,
                                     y_test=y_test, y_pred_on_test=y_pred_on_test)

        return

    def save_model(self, model, weight_file):
        from sklearn.externals import joblib
        joblib.dump(model, weight_file, compress=3)
        return

    def load_model(self, weight_file):
        from sklearn.externals import joblib
        model = joblib.load(weight_file)
        return model

def load_index(number_of_class=3, idx=1):
    filepath = '../data/pics/{}cls_val_ids{}.csv'.format(number_of_class, idx)
    idx_list = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        idx = int(line)
        idx_list.append(idx)
    return idx_list


def index2smiles(idx_list, number_of_class=3):
    print('loading from ../small_model_aliper/data')
    filepath = '../small_model_aliper/data/{}cls_aliper.csv'.format(number_of_class)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines[1:])
    print('{} lines in all'.format(len(lines)))
    idx_list = list(set(idx_list))
    print(idx_list)
    train_idx_list = filter(lambda x:x not in idx_list, range(len(lines)))
    train_lines, test_lines = lines[train_idx_list], lines[idx_list]
    train_smiles_list, train_label_list = line_parser_smiles(train_lines, number_of_class)
    test_smiles_list, test_label_list = line_parser_smiles(test_lines, number_of_class)
    return [train_smiles_list, train_label_list], [test_smiles_list, test_label_list]


def demo_random_forest_classification():
    json_file = 'config/random_forest_classification/{}.json'.format(json_id)
    with open(json_file) as f:
        conf = json.load(f)

    idx_list = load_index(number_of_class, index)

    [train_smiles_list, y_train], [test_smiles_list, y_test] = index2smiles(idx_list, number_of_class)
    x_train, x_test = smiles2fps(train_smiles_list), smiles2fps(test_smiles_list)

    x_train, y_train = x_train.astype(float), y_train.astype(float)
    x_test, y_test = x_test.astype(float), y_test.astype(float)

    print(x_train.shape, '\t', y_train.shape)
    print(x_test.shape, '\t', y_test.shape)

    task = RandomForestClassification(conf=conf)
    task.train_and_predict(x_train, y_train, x_test, y_test, weight_file)
    task.eval_with_existing(x_train, y_train, x_test, y_test, weight_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_file', required=True)
    parser.add_argument('--number_of_class', type=int, default=3)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--json_id', type=int, default=0)

    given_args = parser.parse_args()
    weight_file = given_args.weight_file
    number_of_class = given_args.number_of_class
    index = given_args.index
    json_id = given_args.json_id

    demo_random_forest_classification()
