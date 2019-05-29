from __future__ import print_function

import itertools
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [5, 5]  ### = [10, 10] for 12-class
plt.switch_backend('agg')
from sklearn.metrics import confusion_matrix


oracle = {
    3: ['antineoplastic', 'cardio', 'cns'],
    5: ['antineoplastic', 'gastrointestinal', 'cardio', 'cns', 'antiinfective'],
    12: ['respiratorysystem', 'cns', 'dermatologic', 'urological', 'hematologic', 'antiinflammatory', 'antineoplastic', 'lipidregulating', 'reproductivecontrol', 'antiinfective', 'gastrointestinal', 'cardio']
}


def parser(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if 'train accuracy:' in line:
            line = line.split(':')
            train_acc = float(line[1])
        if 'test accuracy:' in line:
            line = line.split(':')
            test_acc = float(line[1])
    return train_acc, test_acc


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def confusion(mode, num_of_class):
    index = 2
    print(num_of_class)

    filename = 'output_{}/num_class_{}_index_{}.npz'.format(mode, num_of_class, index)
    data = np.load(filename)
    print(data.keys())
    y_true, y_pred = data['y_test'], data['y_pred_on_test']
    print(y_true.shape,'\t', y_pred.shape)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # adj = cnf_matrix.transpoe() / cnf_matrix.sum(axis=1)
    # adj = adj.round(2)

    if num_of_class == 12:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=(5, 5))
    plot_confusion_matrix(cnf_matrix, classes=oracle[num_of_class], normalize=True,
                          title='Confusion matrix')
    plt.savefig('plottings/{}/confusion_num_of_class_{}_index_{}'.format(mode, num_of_class, index),
                bbox_inches='tight')
    return


if __name__ == '__main__':
    mode = 'fingerprints'
    for number_of_class in [3, 5, 12]:
        train_val_list, test_val_list = [], []
        for index in range(0, 5):
            filename = 'output_{}/num_class_{}_index_{}.out'.format(mode, number_of_class, index)
            train_acc, test_acc = parser(filename)
            train_val_list.append(train_acc)
            test_val_list.append(test_acc)
        train_val_list, test_val_list = np.array(train_val_list), np.array(test_val_list)
        print('train mean: {}\tstd: {}'.format(np.mean(train_val_list), np.std(train_val_list)))
        print('test mean: {}\tstd: {}'.format(np.mean(test_val_list), np.std(test_val_list)))
        print()

    confusion(mode='fingerprints', num_of_class=3)
    confusion(mode='fingerprints', num_of_class=5)
    confusion(mode='fingerprints', num_of_class=12)

    mode = 'latent'
    for number_of_class in [3, 5, 12]:
        train_val_list, test_val_list = [], []
        for index in range(1, 6):
            filename = 'output_{}/num_class_{}_index_{}.out'.format(mode, number_of_class, index)
            train_acc, test_acc = parser(filename)
            train_val_list.append(train_acc)
            test_val_list.append(test_acc)
        train_val_list, test_val_list = np.array(train_val_list), np.array(test_val_list)
        print('train mean: {}\tstd: {}'.format(np.mean(train_val_list), np.std(train_val_list)))
        print('test mean: {}\tstd: {}'.format(np.mean(test_val_list), np.std(test_val_list)))
        print()

    confusion(mode='latent', num_of_class=3)
    confusion(mode='latent', num_of_class=5)
    confusion(mode='latent', num_of_class=12)

    confusion(mode='latent_minus4', num_of_class=12)
