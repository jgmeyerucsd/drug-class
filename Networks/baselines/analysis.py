from __future__ import print_function

import numpy as np


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

if __name__ == '__main__':
    for number_of_class in [3, 5, 12]:
        train_val_list, test_val_list = [], []
        for index in range(1, 6):
            filename = 'output/num_class_{}_index_{}.out'.format(number_of_class, index)
            train_acc, test_acc = parser(filename)
            train_val_list.append(train_acc)
            test_val_list.append(test_acc)
        train_val_list, test_val_list = np.array(train_val_list), np.array(test_val_list)
        print('train mean: {}\tstd: {}'.format(np.mean(train_val_list), np.std(train_val_list)))
        print('test mean: {}\tstd: {}'.format(np.mean(test_val_list), np.std(test_val_list)))
        print()