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
    inner_loop_file_path = 'analyze_random_forest_cv_inner_all_single_class.out'
    with open(inner_loop_file_path, 'r') as f:
        lines = f.readlines()

    class_num, running_id, hyper_id = 0, 0, 0
    for line in lines:
        if 'Class Num:' in line:
            print(line.strip())
            line = line.replace(' fold is the test).', '').replace('Top 20 (', '').replace('Class Num: ', '')
            line = line.strip().split('\t')
            class_num, running_id = int(line[0]), int(line[1])
        elif ':' in line:
            line = line.strip().split(':')
            hyper_id = int(line[0])
            outer_file_path = 'output_all_single_class/num_class_{}_id_{}_index_{}.out'.format(class_num, hyper_id, running_id)
            train_acc, test_acc = extract(outer_file_path)
            print('{}: {}'.format(hyper_id, test_acc))
        else:
            print(line.strip())
