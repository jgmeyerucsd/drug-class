from __future__ import print_function

import numpy as np


def extract(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    test_acc, test_balanced_acc, test_matthews_corref, test_roc, test_pr = float('inf'), float('inf'), float('inf'), float('inf'), float('inf')
    for line in lines:
        if 'test accuracy' in line:
            line = line.strip().split(':')
            test_acc = float(line[1])
        if 'test balanced accuracy' in line:
            line = line.strip().split(':')
            test_balanced_acc = float(line[1])
        if 'test matthews corrcoef' in line:
            line = line.strip().split(':')
            test_matthews_corref = float(line[1])
        if 'test multiclass ROC-AUC' in line:
            line = line.strip().split(':')
            test_roc = float(line[1])
        if 'test multiclass PR-AUC' in line:
            line = line.strip().split(':')
            test_pr = float(line[1])
    return test_acc, test_balanced_acc, test_matthews_corref, test_roc, test_pr


if __name__ == '__main__':
    class_num_to_hyper_id = {
        3: [14, 210, 40, 163, 155, 123, 141, 184, 141, 120],
        5: [124, 14, 94, 132, 14, 132, 210, 155, 155, 132],
        12: [9, 57, 88, 72, 88, 141]#, 94, 72, 141, 28]
    }

    for class_num in [3, 5, 12]:
        print('class num: {}'.format(class_num))
        test_acc_list, test_balanced_acc_list, test_matthews_corref_list, test_roc_list, test_pr_list = [], [], [], [], []
        for index, hyper_id in enumerate(class_num_to_hyper_id[class_num]):
            file_path = 'output_small_aliper/num_class_{}_id_{}_index_{}.out'.format(class_num, hyper_id, index)
            test_acc, test_balanced_acc, test_matthews_corref, test_roc, test_pr = extract(file_path)
            test_acc_list.append(test_acc)
            test_balanced_acc_list.append(test_balanced_acc)
            test_matthews_corref_list.append(test_matthews_corref)
            test_roc_list.append(test_roc)
            test_pr_list.append(test_pr)
        print('optimal test accuracy: {} +/- {}'.format(np.mean(test_acc_list), np.std(test_acc_list)))
        print('optimal test balanced accuracy: {} +/- {}'.format(np.mean(test_balanced_acc_list), np.std(test_balanced_acc_list)))
        print('optimal test matthews corrcoef: {} +/- {}'.format(np.mean(test_matthews_corref_list), np.std(test_matthews_corref_list)))
        print('optimal test multiclass ROC-AUC: {} +/- {}'.format(np.mean(test_roc_list), np.std(test_roc_list)))
        print('optimal test multiclass PR-AUC: {} +/- {}'.format(np.mean(test_pr_list), np.std(test_pr_list)))
        print()