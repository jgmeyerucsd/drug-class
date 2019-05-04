from __future__ import print_function

import numpy as np


def extract(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    train_acc, val_acc, test_acc = float('inf'), float('inf'), float('inf')
    for line in lines:
        if 'train accuracy' in line:
            line = line.strip().split(':')
            train_acc = float(line[1])
        if 'val accuracy' in line:
            line = line.strip().split(':')
            val_acc = float(line[1])
        if 'test accuracy' in line:
            line = line.strip().split(':')
            test_acc = float(line[1])
    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    class_num_list = [3, 5, 12]
    K = 20

    for class_num in class_num_list:
        record = {}
        for test_id in range(5):
            record[test_id] = {}
            for hyper_id in range(216):
                record[test_id][hyper_id] = []

        for hyper_id in range(216):
            for idx in range(20):
                test_id = idx / 4
                val_id = idx % 4 + (idx % 4 >= test_id)
                file_path = 'output_fingerprints_cv/num_class_{}_id_{}_index_{}.out'.format(class_num, hyper_id, idx)
                train_acc, val_acc, test_acc = extract(file_path)
                record[test_id][hyper_id].append(val_acc)

        for test_id in range(5):
            for hyper_id in range(216):
                if len(record[test_id][hyper_id]) == 0:
                    record[test_id][hyper_id] = 0
                else:
                    record[test_id][hyper_id] = np.mean(record[test_id][hyper_id])
            
            ordered_record =  sorted(record[test_id].iteritems(), key=lambda (k,v):(v,k), reverse=True)

            print('Class Num: {}\tTop {} ({} fold is the test).'.format(class_num, K, test_id))
            for k,v in ordered_record[:K]:
                print('{}: {}'.format(k, v))
            print()
            print()
        print()
        print()
