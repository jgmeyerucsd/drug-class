#!/usr/bin/env bash

n_list=(3 5 12)
index_list=(0 1 2 3 4 5 6 7 8 9)

mode=aliper
mkdir -p output_"$mode"

for n in "${n_list[@]}"; do
    for index in "${index_list[@]}"; do
        python random_forest_small_aliper.py --weight_file=xxx.pt --number_of_class=$n --index=$index > output_"$mode"/num_class_"$n"_index_"$index".out
    done
done
