
import pandas as pd
import glob
import re

# load data from master table

data_path = "/Users/ijmiller2/Desktop/UW_2019/For_Jesse/drug-class/" + \
    "data/allsingleclass6955_stratkfoldsplits/"
df = pd.read_csv(data_path + "12cls_rmsaltol.csv")

#def index_from_png_path(png_path):
#    return png_path.split("/")[-1]

for file in glob.glob(data_path + "12cls_newskf*.csv"):
    print(file)

k_fold_list = []
for index,row in df.iterrows():
    for file in glob.glob(data_path + "12cls_newskf*.csv"):
        with open(file) as kfold_file:
            for line in kfold_file:
                line = line.rstrip()
                if int(line) == int(index):
                    print(index, file, index)
                    match = re.search('12cls_newskf([0-9]*)', file)
                    k_fold = int(match.group(1)) + 4
                    k_fold_list.append(k_fold)

df['k_fold'] = k_fold_list
df.to_csv("drug_class.txt", sep="\t")

# then edited headers manually
