#!/usr/bin/env bash

n_list=(3 5 12)
index_list=(0 1 2 3 4)

mode=fingerprints

for n in "${n_list[@]}"; do
    for index in "${index_list[@]}"; do
        python random_forest.py --weight_file=xxx.pt --number_of_class=$n --index=$index --mode="$mode" > output_"$mode"/num_class_"$n"_index_"$index".out
    done
done



n_list=(3 5 12)
index_list=(1 2 3 4 5 6 7 8 9 10)

mode=latent

for n in "${n_list[@]}"; do
    for index in "${index_list[@]}"; do
        python random_forest.py --weight_file=xxx.pt --number_of_class=$n --index=$index --mode="$mode" > output_"$mode"/num_class_"$n"_index_"$index".out
    done
done

mode=latent_minus4
n=12
for index in "${index_list[@]}"; do
        python random_forest.py --weight_file=xxx.pt --number_of_class=$n --index=$index --mode="$mode" > output_"$mode"/num_class_"$n"_index_"$index".out
done