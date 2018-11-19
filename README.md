# Code and Data to accompany
# "Learning Molecule Drug Function from Structure Representations and Deep Neural Networks or Random Forests"

### Models

The folder 'MFP_RF' contains the code for the molecular fingerprint + random forests models, and the folder 'IMG_CNN' contains the code for the images and convolutional neural network models. 

### Datasets

All the data is available from the 'data' folder, including the raw lists downloaded from pubchem (data/frompubchem/), the SMILES strings by class (data/SMILES/), the pictures in folders organized by class (data/pics/), and the csv files containing the pngpath, class, and SMILES string (data/pics/12cls_rmsaltol.csv, data/pics/5cls_rmsaltol.csv, data/pics/3cls_rmsaltol.csv). 

### Validation sets

The indexes of validation lines for the 5 validation sets in the above-mentioned example/class list files are in the same folder and named 12cls_val_ids[0-4].csv, 5cls_val_ids[0-4].csv, and 3cls_val_ids[0-4].csv.



