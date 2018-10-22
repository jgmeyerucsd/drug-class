from __future__ import print_function

import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit import Chem
from rdkit.Chem import AllChem

oracle = {
    3: ['antineoplastic', 'cardio', 'cns'],
    5: ['antineoplastic', 'gastrointestinal', 'cardio', 'cns', 'antiinfective'],
    12: ['respiratorysystem', 'cns', 'dermatologic', 'urological', 'hematologic', 'antiinflammatory', 'antineoplastic', 'lipidregulating', 'reproductivecontrol', 'antiinfective', 'gastrointestinal', 'cardio']
}


def load_index(number_of_class=3, idx=1):
    filepath = '../data_no_overlap/pics/{}cls_val_ids{}.csv'.format(number_of_class, idx)
    idx_list = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        idx = int(line)
        idx_list.append(idx)
    return idx_list


def index2smiles(idx_list, number_of_class=3):
    filepath = '../data_no_overlap/pics/{}labels_rmOL_sorted_SMILES.csv'.format(number_of_class)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines[1:])
    idx_list = list(set(idx_list))
    train_idx_list = filter(lambda x:x not in idx_list, range(len(lines)))
    train_lines, test_lines = lines[train_idx_list], lines[idx_list]
    train_smiles_list, train_label_list = line_parser(train_lines, number_of_class)
    test_smiles_list, test_label_list = line_parser(test_lines, number_of_class)
    return [train_smiles_list, train_label_list], [test_smiles_list, test_label_list]


def line_parser(lines, number_of_class):
    smiles_list, label_list = [], []
    for line in lines:
        line = line.strip().split(',')
        label, smiles = line[1], line[2]
        smiles_list.append(smiles)
        label_list.append(oracle[number_of_class].index(label))
    smiles_list = np.array(smiles_list)
    label_list = np.array(label_list)
    return smiles_list, label_list


def smiles2fps(smiles_list):
    fps_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fps = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps = np.array(list(fps.ToBitString()))
        fps_list.append(fps)
    fps_list = np.array(fps_list)
    fps_list = fps_list.astype(np.float)
    return fps_list


if __name__ == '__main__':
    from rdkit.Chem import MolFromSmiles, MolToSmiles
    from rdkit import Chem
    from rdkit.Chem import AllChem

    for n in [3, 5, 12]:
        for idx in range(1, 11):
            idx_list = load_index(n, idx)
            [train_smiles_list, train_label_list], [test_smiles_list, test_label_list] = index2smiles(idx_list, n)
            print(len(train_smiles_list), '\t', len(test_smiles_list))
