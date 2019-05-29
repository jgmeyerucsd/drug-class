### Code and data to accompany:
**[Learning Molecule Drug Function from Structure Representations and Deep Neural Networks or Random Forests](https://doi.org/10.1101/482877)**.
Jesse G Meyer, Shengchao Liu, Ian J Miller, Anthony Gitter, Joshua J Coon.
bioRxiv 2018. doi:10.1101/482877

### Models

The folder 'MFP_RF' contains the code for the molecular fingerprint + random forests models, and the folder 'IMG_CNN' contains the code for the images and convolutional neural network models. 

### Datasets

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


### Validation sets

The indexes of validation lines for the 5 validation sets in the above-mentioned example/class list files are in the same folder and named by the subtask:
1. 12cls_val_ids[0-4].csv
2. 5cls_val_ids[0-4].csv
3. 3cls_val_ids[0-4].csv



