# Code and data to accompany:
**[Learning Drug Functions from Chemical Structures with Convolutional Neural Networks and Random Forests](https://doi.org/10.1021/acs.jcim.9b00236)**.
Jesse G Meyer, Shengchao Liu, Ian J Miller, Joshua J Coon, Anthony Gitter
*Journal of Chemical Information and Modeling*. 2019, 59(10) 4438-4449.

## Models

The folder 'MFP_RF' contains the code for the molecular fingerprint + random forests models, and the folder 'IMG_CNN' contains the code for the images and convolutional neural network models. 

## Data

### Small single-class dataset to mimic Aliper *et al.* 2016 Mol Pharm
All the data is available in the 'small_data_676_chems' folder, including:
1. Notebook with data preparation 'aliper_small_data_prep.ipynb'
2. Lists of each set of molecule SMILES strings split into .csv files for each class "[class]_smiles_rmsalt.csv".
3. Pictures of each molecule in folders by class 'small_data_676_chems/pics'
4. Lists of data for 3, 5, or 12 class problems with the path to their picture, their class, and their SMILES string. These files are used with Fast.ai for training: 3cls_aliper.csv, 5cls_aliper.csv, 12cls_aliper.csv.
5. Files containing the validation set indexes refering to lines in the files described above in #4: '3cls_aliper_10fold[1-9].csv', '5cls_aliper_10fold[1-9].csv', '12cls_aliper_10fold[1-9].csv'

### Full single-class dataset with 6,955 chemicals
All the data is available from the 'data' folder, including:
1. Raw lists downloaded from pubchem (data/frompubchem/), 
2. SMILES strings by class (data/SMILES/)
3. Pictures organized in folders by class (data/pics/)

##### For the IMG+CNN models
Files containing the examples (path to png and SMILES string) and their class annotation are in data/pics/ 
1. data/pics/12cls_rmsaltol.csv
2. data/pics/5cls_rmsaltol.csv
3. data/pics/3cls_rmsaltol.csv

##### For the MFP+RF models
The file containing molecular fingerprints is: data/fingerprints_12cls.csv

##### Validation sets

The indexes of validation lines for the 5 validation sets in the above-mentioned example/class list files are in the same folder and named by the subtask:
1. 12cls_val_ids[0-4].csv
2. 5cls_val_ids[0-4].csv
3. 3cls_val_ids[0-4].csv

### Multi-class data
All multi-class data, including the method to perform multi-class splitting, is available under 'multiclass_data'
1. all_chem_df.csv is the master list with all the info for the chemicals
2. get_data.ipynb is used to get the chemicals from the original downloads with CIDs from pubchem. Also cleans the multiclass data to remove salts and remove repeated chemicals
3. multiclass_5foldCV.ipynb contains the training loop where metrics are computed
4. multiclass_data/pics contains the actual chemical images refered to in 'all_chem_df.csv'
5. the python environment for multiclass data was different and uses fasta version 1. The file fastai_v1.yml contains the environment info needed to recreate
6. as for the other 2 CNN models, the validation indexes refering to lines in all_chem_df.csv are given in multilabel_iter5fold_[0-4].csv
7. multiclass_data/figures gives the network analysis figures

