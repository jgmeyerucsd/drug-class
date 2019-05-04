#!/usr/bin/env bash

n_list=(3 5 12)
n_list=(3)
index_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

mode=fingerprints

for n in "${n_list[@]}"; do
    for index in "${index_list[@]}"; do
        python random_forest_cv.py --weight_file=xxx.pt --number_of_class=$n --index=$index --mode="$mode" >> temp.out
        echo
        echo
        echo
    done
done

