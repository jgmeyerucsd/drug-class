from __future__ import print_function

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef#, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder


def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data

'''
    ## get the roc weighted scores
    log_preds, y = learn.TTA()
    probs = np.mean(np.exp(log_preds),0)
    roc.append(multiclass_roc_auc_score(y, probs, 'weighted'))

    # balanced accuracy
    preds = np.argmax(probs, axis=1)
    bacc.append(balanced_accuracy_score(y, preds))

    # MCC
    mcc.append(matthews_corrcoef(y, preds))
'''


def multiclass_roc_auc_score(y_true, y_prob, AVERAGE='weighted'):
    y_prob = np.mean(y_prob)
    oh = OneHotEncoder(sparse=False, categories='auto')
    yt = oh.fit_transform(y_true.reshape(-1,1))
    return roc_auc_score(yt, y_prob, average=AVERAGE)


def output_classification_result(y_train, y_pred_on_train,
                                 y_val, y_pred_on_val,
                                 y_test, y_pred_on_test):
    if y_train is not None:
        print('train accuracy: {}'.format(accuracy_score(y_true=y_train, y_pred=y_pred_on_train)))
        # print('train multiclass AUC[ROC]: {}'.format(multiclass_roc_auc_score(y_true=y_train, y_prob=y_pred_on_train, AVERAGE='weighted')))
        # print('train balanced accuracy: {}'.format(balanced_accuracy_score(y_true=y_train, y_pred=y_pred_on_train)))
        print('train matthews corrcoef: {}'.format(matthews_corrcoef(y_true=y_train, y_pred=y_pred_on_train)))

    if y_val is not None:
        print('val accuracy: {}'.format(accuracy_score(y_true=y_val, y_pred=y_pred_on_val)))
        # print('val multiclass AUC[ROC]: {}'.format(multiclass_roc_auc_score(y_true=y_val, y_prob=y_pred_on_val, AVERAGE='weighted')))
        print('val balanced accuracy: {}'.format(balanced_accuracy_score(y_true=y_val, y_pred=y_pred_on_val)))
        print('val matthews corrcoef: {}'.format(matthews_corrcoef(y_true=y_val, y_pred=y_pred_on_val)))

    if y_test is not None:
        print('test accuracy: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_on_test)))
        # print('test multiclass AUC[ROC]: {}'.format(multiclass_roc_auc_score(y_true=y_test, y_prob=y_pred_on_test, AVERAGE='weighted')))
        # print('test balanced accuracy: {}'.format(balanced_accuracy_score(y_true=y_test, y_pred=y_pred_on_test)))
        print('test matthews corrcoef: {}'.format(matthews_corrcoef(y_true=y_test, y_pred=y_pred_on_test)))

    return
