# Live Cell Reporter Imaging

---

author:     Nathaniel Evans  
email:      evansna@ohsu.edu

--- 


## Sensitivity analysis use: 

This will run many `single-run analysis` and save the results to disk. 

```{bash} 
$ ./HER2_sensitivity_runs.sh
```
**NOTE**: variables within `HER2_sensitivity_runs.sh` will need to be modified. 

## Single-run analysis use: 

```{bash} 
$ python HER2_classifier.py --data ./HER2_SKBR3_data_6-7-21/ --drug neratinib --sensitive_line WT --resistant_line T798I --load normalized --nclus 15 --out ./output/ --resample_sz 100 --burnin 0
```

`--data` 
---

The outputs of Samuel's processing can be extracted to the necessary file structure using: 
```{bash} 
$ ./HER2_extract_data2.sh 
``` 
**NOTE:** variables within `HER2_extract_data2.sh` will need to be changed for each run. 

directory to data files, should be organized as: 
```
/data_dir/ 
    /dataset_name/ 
        /normalized
            -> clover_all_cell.csv
            -> mscarlet_all_cell.csv
        /raw
            -> clover_all_cell.csv
            -> mscarlet_all_cell.csv
```
        
`--drug`
---
Can be trastuzumab or neratinib

`--sensitive_line` 
---
The cell line to use as sensitive labels [WT]

`--resistant_line` 
---
The cell line to use as resistant labels [T798I, ND611]

`--load` 
---
Whether to use the `normalized` or `raw` data. 

`--nclus`
---
The number of clusters to use. 

`--out`
---
Directory path to save results to

`--resample_sz` 
---
length of time series to resample to 

`--burnin`
---
number of initial time points to ignore in analysis

---

## Concordance Calls 

Use `HER2_sensitivity_results_analysis [<drug>].ipynb` to aggregate the results of the sensitivity analysis, make concordance calls, and save results to file. 

## Conda environment 

```{bash} 
$ conda env create --file environment.yml
```