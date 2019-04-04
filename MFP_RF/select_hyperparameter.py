from __future__ import print_function

import numpy as np


def extract(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    train_acc, test_acc = float('inf'), float('inf')
    for line in lines:
        if 'train accuracy' in line:
            line = line.strip().split(':')
            train_acc = float(line[1])
        if 'test accuracy' in line:
            line = line.strip().split(':')
            test_acc = float(line[1])
    return train_acc, test_acc


if __name__ == '__main__':
    class_num_list = [3, 5, 12]

    for class_num in class_num_list:
        record = {}
        for hyper_id in range(216):
            train_acc_list, test_acc_list = [], []
            for index in range(5):
                file_path = 'output_fingerprints/num_class_{}_id_{}_index_{}.out'.format(class_num, hyper_id, index)
                train_acc, test_acc = extract(file_path)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

            mean_train_acc, mean_test_acc = np.mean(train_acc_list), np.mean(test_acc_list)
            record[hyper_id] = mean_test_acc

        print('Class Num: {}'.format(class_num))
        record = sorted(record.iteritems(), key=lambda (k,v):(v,k), reverse=True)
        for k,v in record:
            print('{}: {}'.format(k, v))
        print()
        print()
        print()
