#!/bin/bash

nburnin="0 5 10"

_repeats_="1 2 3 4 5" 

_nclus_="2 3 4 5 10 15 20"

_resample_sz_="50 75 100 125"

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
                    python HER2_classifier.py --data ./data/HER2/ --drug neratinib --sensitive_line WT --resistant_line T798I --load $load --nclus $nclus --out ./output/ --resample_sz $resample_sz --burnin $burnin
            done
        done 
    done
done 