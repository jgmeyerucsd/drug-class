from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder, label_binarize


def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data


def multiclass_roc_auc_score(y_true, y_pred_proba, AVERAGE='weighted'):
    n_classes = y_pred_proba.shape[1]
    y_true = label_binarize(y_true, classes=[x for x in range(n_classes)])
    return roc_auc_score(y_true, y_pred_proba, average=AVERAGE)


def multiclass_pr_auc_score(y_true, y_pred_proba, AVERAGE='weighted'):
    n_classes = y_pred_proba.shape[1]
    y_true = label_binarize(y_true, classes=[x for x in range(n_classes)])
    return average_precision_score(y_true, y_pred_proba, average=AVERAGE)


def output_classification_result(y_train, y_pred_on_train, y_pred_proba_on_train,
                                 y_val, y_pred_on_val, y_pred_proba_on_val,
                                 y_test, y_pred_on_test, y_pred_proba_on_test):
    if y_train is not None:
        print('train accuracy: {}'.format(accuracy_score(y_true=y_train, y_pred=y_pred_on_train)))
        print('train balanced accuracy: {}'.format(balanced_accuracy_score(y_true=y_train, y_pred=y_pred_on_train)))
        print('train matthews corrcoef: {}'.format(matthews_corrcoef(y_true=y_train, y_pred=y_pred_on_train)))
        print('train multiclass ROC-AUC: {}'.format(multiclass_roc_auc_score(y_true=y_train, y_pred_proba=y_pred_proba_on_train, AVERAGE='weighted')))
        print('train multiclass PR-AUC: {}'.format(multiclass_pr_auc_score(y_true=y_train, y_pred_proba=y_pred_proba_on_train, AVERAGE='weighted')))

    if y_val is not None:
        print('val accuracy: {}'.format(accuracy_score(y_true=y_val, y_pred=y_pred_on_val)))
        print('val balanced accuracy: {}'.format(balanced_accuracy_score(y_true=y_val, y_pred=y_pred_on_val)))
        print('val matthews corrcoef: {}'.format(matthews_corrcoef(y_true=y_val, y_pred=y_pred_on_val)))
        print('val multiclass ROC-AUC: {}'.format(multiclass_roc_auc_score(y_true=y_val, y_pred_proba=y_pred_proba_on_val, AVERAGE='weighted')))
        print('val multiclass PR-AUC: {}'.format(multiclass_pr_auc_score(y_true=y_val, y_pred_proba=y_pred_proba_on_val, AVERAGE='weighted')))

    if y_test is not None:
        print('test accuracy: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_on_test)))
        print('test balanced accuracy: {}'.format(balanced_accuracy_score(y_true=y_test, y_pred=y_pred_on_test)))
        print('test matthews corrcoef: {}'.format(matthews_corrcoef(y_true=y_test, y_pred=y_pred_on_test)))
        print('test multiclass ROC-AUC: {}'.format(multiclass_roc_auc_score(y_true=y_test, y_pred_proba=y_pred_proba_on_test, AVERAGE='weighted')))
        print('test multiclass PR-AUC: {}'.format(multiclass_pr_auc_score(y_true=y_test, y_pred_proba=y_pred_proba_on_test, AVERAGE='weighted')))

    return
