#!/bin/bash

### data paths ### 
input=./data/HER2_SKBR3_data_6-7-21/
output=./neratinib_output/
###################

### nerat/trast ###
drug=neratinib
res_line=T798I
###################


nburnin="0 5 10"

_repeats_="1 2 3 4 5" 

_nclus_="3 5 10 15 20"

_resample_sz_="75 100 125"

_load_="normalized raw" 

for _ in $_repeats_; do
    for load in $_load_; do
        echo $load
        for nclus in $_nclus_; do
            echo -e '\t' $nclus
            for resample_sz in $_resample_sz_; do
                echo -e '\t\t' $resample_sz
                for burnin in $nburnin; do
                    echo -e '\t\t\t' $burnin
                    python HER2_classifier.py --data $input --drug $drug --sensitive_line WT --resistant_line $res_line --load $load --nclus $nclus --out $output --resample_sz $resample_sz --burnin $burnin
                done
            done
        done 
    done
done 
