
import pandas as pd
from rdkit import Chem

df = pd.read_csv("drug_class_test.txt", sep = "\t")

for index,row in df.iterrows():
    smile = row['Canonical_Smiles']
    print(index, Chem.MolFromSmiles(smile))
    # index 2708 produces none

df = df[df['index']!=2708]
# 2708	antiinfective/1169	antiinfective	F[As-](F)(F)(F)(F)F.c1ccc([I+]c2ccccc2)cc1	6

df.to_csv("drug_class_test.txt", sep = "\t")
