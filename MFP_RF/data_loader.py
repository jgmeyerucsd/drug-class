from __future__ import print_function

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit import Chem
from rdkit.Chem import AllChem

oracle = {
    3: ['antineoplastic', 'cardio', 'cns'],
    5: ['antineoplastic', 'gastrointestinal', 'cardio', 'cns', 'antiinfective'],
    12: ['respiratorysystem', 'cns', 'dermatologic', 'urological', 'hematologic', 'antiinflammatory', 'antineoplastic', 'lipidregulating', 'reproductivecontrol', 'antiinfective', 'gastrointestinal', 'cardio']
}


def load_index(number_of_class=3, idx=1):
    filepath = '../data/pics/{}cls_val_ids{}.csv'.format(number_of_class, idx)
    idx_list = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        idx = int(line)
        idx_list.append(idx)
    return idx_list


def index2smiles(idx_list, number_of_class=3):
    filepath = '../data/pics/{}cls_rmsaltol.csv'.format(number_of_class)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines[1:])
    idx_list = list(set(idx_list))
    train_idx_list = filter(lambda x:x not in idx_list, range(len(lines)))
    train_lines, test_lines = lines[train_idx_list], lines[idx_list]
    train_smiles_list, train_label_list = line_parser_smiles(train_lines, number_of_class)
    test_smiles_list, test_label_list = line_parser_smiles(test_lines, number_of_class)
    return [train_smiles_list, train_label_list], [test_smiles_list, test_label_list]


def load_index_valid(number_of_class=3, idx=1):
    def get_list(idx):
        filepath = '../data/pics/{}cls_val_ids{}.csv'.format(number_of_class, idx)
        idx_list = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            idx = int(line)
            idx_list.append(idx)
        return idx_list
    test_id = idx / 4
    val_id = idx % 4 + (idx % 4 >= test_id)
    print('test id: {}\t val id: {}'.format(test_id, val_id))
    val_list, test_list = get_list(val_id), get_list(test_id)
    return val_list, test_list


def index2smiles_valid(val_idx_list, test_idx_list, number_of_class=3):
    filepath = '../data/pics/{}cls_rmsaltol.csv'.format(number_of_class)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines[1:])
    idx_list = list(set(val_idx_list + test_idx_list))
    train_idx_list = filter(lambda x:x not in idx_list, range(len(lines)))
    train_lines, val_lines, test_lines = lines[train_idx_list], lines[val_idx_list], lines[test_idx_list]

    train_smiles_list, train_label_list = line_parser_smiles(train_lines, number_of_class)
    val_smiles_list, val_label_list = line_parser_smiles(val_lines, number_of_class)
    test_smiles_list, test_label_list = line_parser_smiles(test_lines, number_of_class)
    return [train_smiles_list, train_label_list], [val_smiles_list, val_label_list], [test_smiles_list, test_label_list]


def index2latent(idx_list, number_of_class=3):
    filepath = '../data_no_overlap/latents_{}cls_v2_sorted.csv'.format(number_of_class)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines[1:])
    idx_list = list(set(idx_list))
    train_idx_list = filter(lambda x:x not in idx_list, range(len(lines)))
    train_lines, test_lines = lines[train_idx_list], lines[idx_list]
    train_latent_list, train_label_list = line_parser_latent(train_lines, number_of_class)
    test_latent_list, test_label_list = line_parser_latent(test_lines, number_of_class)
    return [train_latent_list, train_label_list], [test_latent_list, test_label_list]


def index2latent_minus4(idx_list, number_of_class=3):
    filepath = '../data_no_overlap/latent_{}cls_layerminus4_sorted.csv'.format(number_of_class)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines[1:])
    idx_list = list(set(idx_list))
    train_idx_list = filter(lambda x:x not in idx_list, range(len(lines)))
    train_lines, test_lines = lines[train_idx_list], lines[idx_list]
    train_latent_list, train_label_list = line_parser_latent(train_lines, number_of_class)
    test_latent_list, test_label_list = line_parser_latent(test_lines, number_of_class)
    return [train_latent_list, train_label_list], [test_latent_list, test_label_list]


def line_parser_smiles(lines, number_of_class):
    smiles_list, label_list = [], []
    for line in lines:
        line = line.strip().split(',')
        label, smiles = line[1], line[2]
        smiles_list.append(smiles)
        label_list.append(oracle[number_of_class].index(label))
    smiles_list = np.array(smiles_list)
    label_list = np.array(label_list)
    return smiles_list, label_list


def line_parser_latent(lines, number_of_class):
    latent_list, label_list = [], []
    for line in lines:
        line = line.strip().split(',')
        label, latent = line[1], line[3:]
        latent_list.append(latent)
        label_list.append(oracle[number_of_class].index(label))
    latent_list = np.array(latent_list)
    latent_list.astype(float)
    label_list = np.array(label_list)
    return latent_list, label_list


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
        for idx in range(0, 5):
            idx_list = load_index(n, idx)
            [train_smiles_list, train_label_list], [test_smiles_list, test_label_list] = index2smiles(idx_list, n)
            print(len(train_smiles_list), '\t', len(test_smiles_list))
    print()

    # Put Fingerprints into csv files
    for n in [3, 5, 12]:
        with open('../data/pics/{}cls_rmsaltol.csv'.format(n), 'r') as f:
            lines = f.readlines()
        lines = lines[1:]
        idx = range(len(lines))
        [temp_smiles_list, temp_list], [smiles_list, label_list] = index2smiles(idx, number_of_class=n)
        fps_array = smiles2fps(smiles_list).astype(int).astype(str)
        fps_list = []
        for i in range(len(fps_array)):
            fps_list.append(''.join(fps_array[i]))
        print(len(fps_list), '\t', len(label_list))
        df = pd.DataFrame({'Fingerprints': fps_list, 'label': label_list})
        df.to_csv('../data/fingerprints_{}cls.csv.gz'.format(n), index=None, compression='gzip')

    # check if two files match
    n = 12
    filepath_smiles = '../data_no_overlap/pics/{}labels_rmOL_sorted_SMILES.csv'.format(n)
    filepath_latent = '../data_no_overlap/{}cls_ordered_512latents.csv'.format(n)

    with open(filepath_smiles, 'r') as f:
        lines_smiles = f.readlines()
    with open(filepath_latent, 'r') as f:
        lines_latent = f.readlines()

    for a,b in zip(lines_smiles, lines_latent):
        a = a.strip().split(',')
        b = b.strip().split(',')

        if a[1] != b[1]:
            print(a[1], '\t', b[1], '\t')
