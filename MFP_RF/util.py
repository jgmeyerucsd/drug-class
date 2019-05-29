from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score


def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data


def output_classification_result(y_train, y_pred_on_train,
                                 y_val, y_pred_on_val,
                                 y_test, y_pred_on_test):
    print('train accuracy: {}'.format(accuracy_score(y_true=y_train, y_pred=y_pred_on_train)))
    if y_val is not None:
        print('val accuracy: {}'.format(accuracy_score(y_true=y_val, y_pred=y_pred_on_val)))
    if y_test is not None:
        print('test accuracy: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_on_test)))
    return

