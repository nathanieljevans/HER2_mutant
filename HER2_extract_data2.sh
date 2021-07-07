#!/bin/bash

# 
# pull out the specific files we need for the HER2 project from all of samules results 
#
# ./extract_data.sh
# 
# 

inp="/mnt/z/Nate/Live cell reporter imaging/HER2 project/20210604_New/"
out=/mnt/c/Users/evansna/Documents/HER2_SKBR3_data/

echo 'starting' 

rm -r "$out"/*

echo 'deleted output dir files' 

for f in "$inp"/*; do
	if [ -d "$f" ]; then
		# Will not run if no directories are available
		echo "$f"
		mydir=`basename "$f"`
		echo $mydir

		if [ -d "$f"/output_abs_sys_corr/ ]; then
			mkdir $out/$mydir
			mkdir $out/$mydir/raw/
			mkdir $out/$mydir/normalized/
			cp "$inp"/$mydir/output_abs_sys_corr/clover/clover_all_cell.csv $out/$mydir/normalized/clover_all_cell.csv
			cp "$inp"/$mydir/output_abs_sys_corr/mscarlet/mscarlet_all_cell.csv $out/$mydir/normalized/mscarlet_all_cell.csv

			cp "$inp"/$mydir/output_abs/clover/clover_all_cell.csv $out/$mydir/raw/clover_all_cell.csv
			cp "$inp"/$mydir/output_abs/mscarlet/mscarlet_all_cell.csv $out/$mydir/raw/mscarlet_all_cell.csv
		fi
	fi
done
