# Results Data Dictionary 

## mutant 
The HER2 mutation identifier specific to a given set of data. One of the group keys (mutant, treatment, cell_line)

## no-call 
The number of `single-run analysis` (e.g., one set of hyper-parameters) that assigned the mutant to "no-call" group. Note: `no-call`, `res` and `sens` should sum to the number of `single-run analysis` performed in the sensitivity analysis (e.g., 90 hyperparms tested * 5 replicates = 450), unless a mutant was run more than once. 

## res
The number of `single-run analysis` (e.g., one set of hyper-parameters) that assigned the mutant to "res" group. 

## sens 
The number of `single-run analysis` (e.g., one set of hyper-parameters) that assigned the mutant to "sens" group. 

## prop_call_res
The proportion of calls made as resistant for each respective mutant. Calcualted by   
$$ res / (res + sens + nocall) $$  

## prop_call_sens

See `prop_call_res`. 

## prop_call_nocall

See `prop_call_res`. 

## cell_track_counts

The number of cell tracks that were used in analysis of each mutant. If a mutant was run multiple times, then the first `cell_track_count` is listed here, e.g., be wary of this column for controls (ND611, T798I) or multi-run mutants. 


## low_data_flag 

A flag was is set if the `cell_track_counts` are less than 5% of the median `cell_track_counts`. As with `cell_track_counts`, be wary of this column for multi-run mutants.


## prop_PC<xx>_batch_flag (xx = 1,2)

The proportion of `single-run analysis` that had a PC<xx> batch flag set. e.g., a value of 0.15 in this column indicates that this mutant had a batch effect flag set for 15% of the hyper-parameters tested. Recommend end-user be wary of mutants with high values. 
