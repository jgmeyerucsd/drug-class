
import pubchempy as pcp
import glob
import pandas as pd

glob_path = "/Users/ijmiller2/Desktop/UW_2018/For_Jesse/drug-class/Networks/data/fromPubChem/*txt"

tables_list=glob.glob(glob_path)
print(tables_list)
CID_dict = {}
cidslist = []
for table in tables_list:
    #tableID = table.replace('.txt','')
    #tableID = tableID.replace('../data/','')
    tableID = table.split("/")[-1].rstrip(".txt")
    CID_dict[tableID] = []
    with open(table) as inf:
        for aline in inf.readlines():
            if aline.startswith('CID')==True:
                #cid = aline.replace('\n')
                CID_dict[tableID].append(aline.replace('\n','').replace('CID: ',''))


print(len(CID_dict[tableID]))
CID_dict.keys()
print(CID_dict[tableID][0])

#get SMILES for CIDs
#keep things with smiles length <400

smiles_dict = {}

for key in CID_dict.keys():
    smiles_dict[key] = []
    prop_dict = pcp.get_properties('IsomericSMILES', CID_dict[key])
    for i in range(0, len(prop_dict)):
        if len(prop_dict[i]['IsomericSMILES'])<400:  #### only those under 200 char
            smiles_dict[key].append(prop_dict[i]['IsomericSMILES'])
            cidslist.append(i)


print(prop_dict[i]["IsomericSMILES"])
print(len(smiles_dict[key]))

p = pcp.get_properties('IsomericSMILES', 'CC', 'smiles', searchtype='superstructure')
p = pcp.get_properties('C[N+](C)(C)CC(=O)[O-]', 'CC', 'smiles', searchtype='superstructure')

property_list = ['MolecularWeight', 'HBondDonorCount', 'HBondAcceptorCount', 'FeatureHydrophobeCount3D', 'IsomericSMILES']
#CID_identifier = 'C[N+](C)(C)CC(=O)[O-]'
CID_identifier = [630, 631]
p = pcp.get_properties(property_list, CID_identifier, as_dataframe=True)

#This list is 12061
full_cidlist = [cid for cids in CID_dict.values() for cid in cids]
p = pcp.get_properties(property_list, full_cidlist, as_dataframe=True)

drug_class_for_df = []
for CID,row in p.iterrows():
    drug_classes = []
    for drug_class,CID_list in CID_dict.items():
        if str(CID) in CID_list:
            print("{} in {}".format(CID,drug_class))
            drug_classes.append(drug_class)
    if len(drug_classes) > 1:
        drug_class_for_df.append("multi")
    else:
        drug_class_for_df.append(drug_classes[0])
    #print("\t".join([str(CID)] + drug_classes))

p['drug_class'] = drug_class_for_df
p.to_csv("CID_properties.csv", na_rep="NA")
