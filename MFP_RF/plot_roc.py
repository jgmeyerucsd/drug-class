from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np


if __name__ == '__main__':
    class_num_to_hyper_id = {
        3: [12, 12, 38, 57, 155],
        5: [65, 139, 148, 148, 210],
        12: [124, 65, 148, 88, 148]
    }

    index_to_label = {
        3: ['antineoplastic', 'cardio', 'cns'],
        5: ['antiinfective', 'antineoplastic', 'cardio', 'cns', 'gastrointestinal'],
        12: ['antiinfective', 'antiinflammatory', 'antineoplastic', 'cardio', 'cns', 'dermatologic',
             'gastrointestinal', 'hematologic', 'lipidregulating', 'reproductivecontrol', 'respiratorysystem', 'urological']
    }

    for class_num in [3, 5, 12]:
        data_file = 'output_all_single_class/num_class_{}_index_4.npz'.format(class_num)
        data = np.load(data_file)
        print(data.keys())
        y_test = data['y_test']
        y_pred_proba_on_test = data['y_pred_proba_on_test']

        for x in range(class_num):
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_on_test[:,x], pos_label=x)
            plt.plot(fpr, tpr, label=index_to_label[class_num][x])
        plt.legend()
        plt.savefig('plottings/{}_roc_curve.svg'.format(class_num), type='svg')
        plt.clf()
