'''
HER2 project, led by Dr. Samuel Tsang

Summary: Goal is to identify sensitive and resistant mutant lines. 

author: Nathaniel Evans
email: evansna@ohsu.edu

---

Use: 

```{bash} 

$ python HER2_classifier.py --data ./HER2_SKBR3_data_6-7-21/ --drug neratinib --sensitive_line WT --resistant_line T798I --load normalized --nclus 15 --out ./output/ --resample_sz 100 --burnin 0
```

`--data` 
directory to data files, should be organized as: 
/data/ 
    /HER2/ 
        /dataset_name/ 
            /normalized
                -> clover_all_cell.csv
                -> mscarlet_all_cell.csv
            /raw
                -> clover_all_cell.csv
                -> mscarlet_all_cell.csv
            
`--drug`
Can be trastuzumab or neratinib

`--sensitive_line` 
The cell line to use as sensitive labels 

`--resistant_line` 
The cell line to use as resistant labels

`--load` 
Whether to use the `normalized` or `raw` data. 

`--nclus`
The number of clusters to use. 

`--out`
Directory path to save results to

`--resample_sz` 
length of time series to resample to 

`--burnin`
number of initial time points to ignore in analysis

---
'''

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sbn

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans
from tslearn.metrics import dtw

from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA
from sklearn.svm import SVC

import argparse
from datetime import datetime

import sys 
import uuid
import os

__VERSION_ID__ = '1.1'

def get_args(): 
    parser = argparse.ArgumentParser(description='Identify sensitive and resistant mutant lines. ')
    
    parser.add_argument('--data', type=str, nargs=1,
                        help='data directory path')
    
    parser.add_argument('--out', type=str, nargs=1,
                        help='output directory path')
    
    parser.add_argument('--drug', type=str, nargs=1,
                        help='drug to use (can be Neratinib or Trastuzumab)')
    
    parser.add_argument('--sensitive_line', type=str, nargs=1,
                        help='cell line to use for sensitive labels')
    
    parser.add_argument('--resistant_line', type=str, nargs=1,
                        help='cell line to use for resistant labels')
    
    parser.add_argument('--load', type=str, nargs=1,
                        help='whether to use `normalized` or `raw` data.')
    
    parser.add_argument('--nclus', type=int, nargs=1,
                        help='number of clusters to use')
    
    parser.add_argument('--resample_sz', type=int, nargs=1,
                        help='length of time series to resample to')

    parser.add_argument('--burnin', type=int, nargs=1,
                        help='number of early time points to remove from analysis')
    
    args = parser.parse_args()
    
    assert args.drug[0].lower() in ['neratinib', 'trastuzumab'], '`--drug` must be either "trastuzumab" or "neratinib"' 
    assert args.load[0].lower() in ['normalized', 'raw'], '`--load` must be either "normalized" or "raw"'
    assert (args.nclus[0] <= 50) & (args.nclus[0] >= 2), '`--nclus` should be an integer between 2 and 50'    
    assert (args.resample_sz[0] <= 150) & (args.resample_sz[0] >= 25), '`--resample_sz` should be an integer between 25 and 150' 
    
    return args


def load_data(args): 
    '''

    '''
    print('\nloading data...')
    load = args.load[0].lower()
    data_dir = args.data[0]
    datasets = [x for x in os.listdir(data_dir) if os.path.isdir(data_dir + x)]
    
    print('# of datasets to load:', len(datasets))
    
    series_sel = pd.read_csv(data_dir + datasets[0] + '/' + load + '/clover_all_cell.csv').columns[1:-3]

    _datas = []
    for dataset in datasets: 
        cl_path = data_dir + dataset + '/' + load + '/clover_all_cell.csv'
        ms_path = data_dir + dataset + '/' + load + '/mscarlet_all_cell.csv'
        _clover = pd.read_csv(cl_path)
        _mscarl = pd.read_csv(ms_path)
        _data = _clover.merge(_mscarl, on=['track_index', 'cell__treatment'], how='inner')
        _data = _data.assign(dataset=dataset)
        _datas.append(_data)

    data = pd.concat(_datas, axis=0)

    clover_sel = [f'{x}_x' for x in series_sel]
    mscarl_sel = [f'{x}_y' for x in series_sel]

    data.cell__treatment = data.cell__treatment.str.upper()
    data = data.assign(drug = [x.split('_', maxsplit=5)[-1].lower() for x in data.cell__treatment])
    data = data.assign(cell_line = [x.split('_', maxsplit=5)[0].upper() for x in data.cell__treatment])
    data = data.assign(mutant = [x.split('_', maxsplit=5)[-2].upper() for x in data.cell__treatment])

    print('mapping drug names to one name...')
    drug_map = {'untreated':'untreated', 
                '10nm_neratinib':'10nm_neratinib', 
                '10ug_ml_trastuzumab':'10ug_ml_trastuzumab', 
                'neratinib':'10nm_neratinib',
                'trastuzumab':'10ug_ml_trastuzumab'}

    data = data.assign(drug = lambda x: x.drug.map(drug_map))

    return data, clover_sel, mscarl_sel

def filter_na(data, args, clover_sel, mscarl_sel): 
    '''

    '''
    print('\nfiltering to drug and removing NAs...')
    
    if args.drug[0].lower() == 'neratinib': 
        drug_ = '10nm_neratinib'
    else: 
        drug_ = '10ug_ml_trastuzumab'
            
    data = data[lambda x: x.drug.isin(['untreated', drug_])]
    print('Data shape (untreated + drug):', data.shape)
    
    print('length of time series BEFORE removing time points with NA', len(clover_sel))
    clover_sel = np.array(clover_sel)[~data[clover_sel].isna().any()]
    mscarl_sel = np.array(mscarl_sel)[~data[mscarl_sel].isna().any()]
    assert len(clover_sel) == len(mscarl_sel), 'clover timeseries is different length than mscarlet time series'
    print('length of time series AFTER removing time points with NA', len(clover_sel))

    low_data_flags = data.groupby(['mutant', 'drug']).count()['track_index'].reset_index().rename({'track_index':'cell_track_count'}, axis=1).assign(low_data_flag = lambda x: x.cell_track_count < (x.cell_track_count.median() / 10)) 
    low_data_flags = low_data_flags.assign(drug = [x.split('_')[-1] if '_' in x else x for x in low_data_flags.drug.values])

    return data, low_data_flags, clover_sel, mscarl_sel

def resample(data, args, clover_sel, mscarl_sel): 
    '''

    '''
    print('\nresampling time series...')
        
    X_train = np.stack([data[clover_sel], data[mscarl_sel]], axis=2)
    print('Training data shape BEFORE resampling:', X_train.shape)

    # Make time series shorter
    X_train = TimeSeriesResampler(sz=args.resample_sz[0]).fit_transform(X_train)
    print('Training data shape AFTER resampling:', X_train.shape)

    return X_train

def burnin(args, clover_sel, mscarl_sel):
    '''
    '''
    i = args.burnin[0]
    print('adding burnin of', i)
    csel = clover_sel[i:]
    msel = mscarl_sel[i:]
    return csel, msel

def fit_timeseries_kmeans(args, X_train, plot=True, save=None): 
    '''
    '''
    print('\nperforming time-series kmeans clustering...')
    print()
    km = TimeSeriesKMeans(n_clusters=args.nclus[0], verbose=True,metric='euclidean', n_jobs=8)
    y_pred = km.fit_predict(X_train)
    print()
    
    if plot:
        F = plt.figure(figsize=(20,10))
        for yi in range(args.nclus[0]):
            if args.nclus[0] % 5 == 0: 
                nrows = int(args.nclus[0] / 5) 
            else: 
                nrows = int(args.nclus[0] / 5)  + 1
                
            plt.subplot(nrows, 5, yi + 1)
            for xx in X_train[y_pred == yi][0:250]:
                plt.plot(xx[:,0], "r-", alpha=.05)
                plt.plot(xx[:,1], "b-", alpha=.05)

            plt.title(f'cluster sz: {len(X_train[y_pred == yi])}')
            plt.plot(km.cluster_centers_[yi][:,0], "r-", label='clover')
            plt.plot(km.cluster_centers_[yi][:,1], "b-", label='mscarlet')

            plt.xlim(0, args.resample_sz[0])
            plt.ylim(0, 1)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)

        plt.tight_layout()
        
        if save is not None: 
            plt.savefig(save + '/cluster_plots.png')
        else: 
            plt.show()

        plt.close('all')

    return y_pred, km

def quantify_cluster_prop(args, data, y_pred): 
    '''
    '''
    print('\nquantifying experiment by cluster proportions...')
    lb = LabelEncoder()
    y_trt = lb.fit_transform([f'{x}--{y}--{z}' for x,y,z in zip(data.drug.values, data.mutant.values, data.dataset.values)])

    cm_cnts = {c:np.zeros(args.nclus[0]) for c in lb.classes_} 

    for i, clus, grp in zip(range(len(y_pred)), y_pred, y_trt) :
        cm_cnts[lb.classes_[grp]][clus] += 1

    cm_prob = {k:v/np.sum(v) for k,v in cm_cnts.items()}

    labels = [k for k,v in cm_prob.items()]
    cm = np.stack([v for k,v in cm_prob.items()], axis=0)

    return cm, lb

def plot_cluster_corr(cm, save=None):
    '''
    '''
    corr = np.corrcoef(cm, rowvar=False)

    f = plt.figure(figsize=(7,7))
    ax = sbn.clustermap(
        corr, 
        vmin=-1, vmax=1, center=0,
    )
    
    if save is not None: 
        plt.savefig(save + '/cluster_corr_plot.png')
    else: 
        plt.show()

    plt.close('all')

def plot_cluster_heatmap(cm, lb, save=None): 
    labels = lb.classes_
    drug = [x.split('--')[0] for x in labels]
    lut = dict(zip(set(drug), sbn.hls_palette(len(set(drug)), l=0.5, s=0.8)))
    row_colors = pd.DataFrame(drug)[0].map(lut)

    #Create additional row_colors here
    cell_line = [x.split('--')[1] for x in labels]
    lut2 = dict(zip(set(cell_line), sbn.hls_palette(len(set(cell_line)), l=0.5, s=0.8)))
    row_colors2 = pd.DataFrame(cell_line)[0].map(lut2)

    df = pd.DataFrame(index=labels, data=cm)
    sbn.clustermap(df, figsize=(12,15), row_colors=[row_colors, row_colors2]) 

    plt.ylabel('cluster membership')
    
    if save is not None: 
        plt.savefig(save + '/cluster_heatmap.png')
    else: 
        plt.show()

def check_batch_effects(args, res, plot=True, save=None): 
    '''
    '''
    _sens = args.sensitive_line[0].upper()
    _res = args.resistant_line[0].upper()

    batcheffect = res[lambda x: x.cell_line.isin([_sens, _res])]

    y_pc1= batcheffect[['pc1']].values.astype(float, copy=True)
    y_pc2= batcheffect[['pc2']].values.astype(float, copy=True)

    print(res.exp_set.unique())
    lb_exp = OneHotEncoder() ; lb_treat = OneHotEncoder() ; lb_line = OneHotEncoder()
    X_exp = lb_exp.fit_transform( batcheffect['exp_set'].values.reshape(-1,1) ).toarray()
    X_treat = lb_treat.fit_transform( batcheffect['treatment'].values.reshape(-1,1) ).toarray()
    X_line = lb_line.fit_transform( batcheffect['cell_line'].values.reshape(-1,1) ).toarray()

    feat_order = lb_exp.categories_[0].tolist() + lb_treat.categories_[0].tolist() + lb_line.categories_[0].tolist()
    print(feat_order)
    print('feature order:')
    _ = [print('\t', f'x{i+1}', ' -> ', f) for i,f in enumerate(feat_order)]

    X = sm.add_constant( np.concatenate([X_exp, X_treat, X_line], axis=1) ) 

    print('---'*25)
    print('PC1 ANOVA')
    print('---'*25)
    lm = sm.OLS(y_pc1, X)
    lm_res = lm.fit() 
    print( lm_res.summary() )
    PC1_pvals = pd.DataFrame({x:y for x,y in zip(['constant'] + feat_order, lm_res.pvalues)}, index=[0]).assign(PC = 1)

    print('---'*25)
    print('PC2 ANOVA')
    print('---'*25)
    lm = sm.OLS(y_pc2, X)
    lm_res = lm.fit() 
    print( lm_res.summary() )
    PC2_pvals = pd.DataFrame({x:y for x,y in zip(['constant'] + feat_order, lm_res.pvalues)}, index=[0]).assign(PC = 2)

    batch_res = pd.concat([PC1_pvals, PC2_pvals], ignore_index=True, axis=0)

    if plot: 
        f, axes= plt.subplots(1,2, figsize=(20,5))
        sbn.boxplot(x='exp_set', y='pc1', data=batcheffect, ax=axes[0])
        sbn.boxplot(x='exp_set', y='pc2', data=batcheffect, ax=axes[1])

        sbn.scatterplot(x='exp_set', y='pc1', data=batcheffect, ax=axes[0])
        sbn.scatterplot(x='exp_set', y='pc2', data=batcheffect, ax=axes[1])

        for tick in axes[0].get_xticklabels():
            tick.set_rotation(45)

        for tick in axes[1].get_xticklabels():
            tick.set_rotation(45)

        plt.tight_layout()
        
        if save is not None: 
            plt.savefig(save + '/controls_by_batch.png')
        else: 
            plt.show()

        f, axes= plt.subplots(1,2, figsize=(20,5))
        sbn.boxplot(x='treatment', y='pc1', hue='exp_set', data=batcheffect, ax=axes[0])
        sbn.boxplot(x='treatment', y='pc2', hue='exp_set', data=batcheffect, ax=axes[1])
        plt.tight_layout()

        for tick in axes[0].get_xticklabels():
            tick.set_rotation(45)

        for tick in axes[1].get_xticklabels():
            tick.set_rotation(45)

        if save is not None: 
            plt.savefig(save + '/controls_by_batch_and_treatment.png')
        else: 
            plt.show()

    return batch_res

def reduce_dim(args, cm, lb, plot=True, save=None): 
    _sens = args.sensitive_line[0].upper()
    _res = args.resistant_line[0].upper()
    _drug = args.drug[0].lower()
    
    print('\nperforming dim. reduction (pca)...')
    pca = PCA(n_components=2)
    PCs = pca.fit_transform(cm)

    print('PCA explained variance ratio:', pca.explained_variance_ratio_)
    print('PC shape:', PCs.shape)
    
    res = pd.DataFrame({'pc1': PCs[:,0], 'pc2':PCs[:,1], 'treatment':[x.split('--')[0].lower().split('_')[-1] for x in lb.classes_], 'cell_line':[x.split('--')[1].upper() for x in lb.classes_], 'exp_set':[x.split('--')[-1] for x in lb.classes_]})
    
    if plot: 
        plt.figure(figsize=(12,12))
        sbn.scatterplot(x='pc1', y='pc2', data=res, hue='cell_line', style='treatment', s=300)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=5)

        if save is not None:
            plt.savefig(save + '/PCA_all.png', bbox_inches='tight')
        else: 
            plt.show()
        
        plt.figure(figsize=(12,12))
        sbn.scatterplot(x='pc1', y='pc2', data=res[lambda x: (x.cell_line.isin([_sens, _res]))], hue='cell_line', style='treatment', s=300)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)
        
        if save is not None: 
            plt.savefig(save + '/PCA_labeled.png',bbox_inches='tight')
        else: 
            plt.show()

        plt.close('all')

    return res, pca

def train_classifier(args, res, plot=True, save=None): 
    '''
    '''
    _sens = args.sensitive_line[0].upper()
    _res = args.resistant_line[0].upper()
    _drug = args.drug[0].lower()

    print('\ntraining classifier...')
        
    print('sensitive line: \t', _sens)
    print('resistant line: \t', _res)
    print('drug:\t\t', _drug)
    
    res_drug = res[lambda x: (x.cell_line.isin([_sens, _res])) & (x.treatment == _drug)]
    print('drug + WT df size: ', res_drug.shape)

    X = res_drug[['pc1', 'pc2']].values
    y_res = ((res_drug.cell_line == _res).values)
    y_sens = ((res_drug.cell_line == _sens).values)

    assert (y_res == ~y_sens).all(), 'y class label assignment has more than 2 classes...'
    
    y = 1*y_sens
    
    print('X train shape:', X.shape)
    print('# neg class (resistant):', np.sum(y_res))
    print('# pos class (sensitive):', np.sum(y_sens))
        
    model = SVC(kernel='rbf', C=1, gamma='scale', probability=True, tol=1e-6)
    model.fit(X,y) 
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    if plot: 
        plt.figure(figsize=(10, 5))
        plt.subplots_adjust(bottom=.2, top=.95)
        
        xlims = (min(X[:,0]), max(X[:,0]))
        ylims = (min(X[:,1]), max(X[:,1]))
        xpad = 1.5*(xlims[1] - xlims[0]) / 2
        ypad = 1.5*(ylims[1] - ylims[0]) / 2
        xlims = (min(X[:,0]) - xpad, max(X[:,0]) + xpad)
        ylims = (min(X[:,1]) - ypad, max(X[:,1]) + ypad)

        xx = np.linspace(*xlims, 100)
        yy = np.linspace(*ylims, 100).T
        xx, yy = np.meshgrid(xx, yy)
        Xfull = np.c_[xx.ravel(), yy.ravel()]
        
        # View probabilities:
        probas = model.predict_proba(Xfull)
        n_classes = np.unique(y_pred).size
        class_names = ['resistant', 'sensitive']
        name = 'Support Vector Classifier [RBF kernel]'

        for k in range(n_classes):
            plt.subplot(1, n_classes, 0 * n_classes + k + 1)
            plt.title("%s class" % class_names[k])
            if k == 0:
                plt.ylabel(name)
            imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                        extent=(*xlims, *ylims), origin='lower')
            plt.xticks(())
            plt.yticks(())
            idx = (y_pred == k)
            if idx.any():
                plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='w', edgecolor='k')
        
        ax = plt.axes([0.15, 0.04, 0.7, 0.05])
        plt.title("Probability")
        plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')

        if save is not None: 
            plt.savefig(save + '/classifier_results.png', bbox_inches='tight')
        else: 
            plt.show() 

        plt.close('all')

    return model, accuracy 

def predict_mutants(args, model, res, batch_res, low_data_flags):
    '''
    '''
    print('\npredicting unlabeled sensitivities...')
    _other = res[lambda x: ~(x.cell_line.isin([args.sensitive_line[0], args.resistant_line[0]])) & (x.treatment == args.drug[0].lower())].reset_index(drop=True)

    X_all = _other[['pc1', 'pc2']].values

    y_hat = model.predict_proba(X_all)

    pres = pd.DataFrame({'prob_res':y_hat[:,0], 'prob_sens':y_hat[:,1]})
    prob_res = pd.concat([_other, pres], axis=1) 

    # assign calls 
    prob_res = prob_res.assign(call=[['res','sens'][np.argmax([x,y])] if (np.abs(np.log2(x/y)) > 0.6) else 'no-call' for x,y in zip(prob_res.prob_res, prob_res.prob_sens)]) # no call if odds ratio < ~1.5

    # merge with batch effect results
    pc1_batch = batch_res[lambda x: x.PC == 1][lambda x: x.columns[1:-5]].unstack().reset_index().rename({'level_0':'batch', 0:'PC1_batch_pval'}, axis=1).drop('level_1', axis=1).assign(PC1_batch_flag= lambda x: x.PC1_batch_pval < 0.05)
    pc2_batch = batch_res[lambda x: x.PC == 2][lambda x: x.columns[1:-5]].unstack().reset_index().rename({'level_0':'batch', 0:'PC2_batch_pval'}, axis=1).drop('level_1', axis=1).assign(PC2_batch_flag= lambda x: x.PC2_batch_pval < 0.05)
    batch_calls = pc1_batch.merge(pc2_batch, on='batch')

    prob_res2 = prob_res.assign(odds_ratio = lambda x: x.prob_res / x.prob_sens).merge(batch_calls, left_on='exp_set', right_on='batch', how='left').sort_values(by='odds_ratio')

    prob_res3 = prob_res2.merge(low_data_flags, left_on=['cell_line', 'treatment'], right_on=['mutant', 'drug'], how='left')

    prob_res3 = prob_res3.drop(['cell_line'], axis=1)

    return prob_res3

if __name__ == '__main__': 
          
    args = get_args() 
          
    run_id = str(uuid.uuid4())    
          
    output_dir = args.out[0] + '/' + run_id
    
    os.mkdir(output_dir)
    out_log = output_dir + '/console_output.log'
    print('-'*25)
    print(args)
    print('console output will be logged to:', out_log)
    print('-'*25)
    
    with open(out_log, 'w') as sys.stdout: 
        
        print('config...')
        print('-'*25)
        print('script version:', __VERSION_ID__)
        print(args)
        print()
        print(datetime.now())
        print('-'*25)
        
        ########### LOAD DATA ###########################################################
    
        data, clover_sel, mscarl_sel = load_data(args)
        
        ########### FILTER TO DRUG & REMOVE NA ##########################################

        data, low_data_flags, clover_sel, mscarl_sel = filter_na(data, args, clover_sel, mscarl_sel)

        ########### REMOVE EARLY TIME POINTS "BURN IN" ##################################

        clover_sel, mascarl_sel = burnin(args, clover_sel, mscarl_sel)

        ########### RESAMPLE ############################################################

        X_train = resample(data, args, clover_sel, mscarl_sel)
        
        ########### FIT TIME SERIES K-MEANS #############################################
        
        y_pred, km = fit_timeseries_kmeans(args, X_train, plot=True, save=output_dir)

        ########### CALCULATE CLUST PROPORTIONS #########################################

        cm, lb = quantify_cluster_prop(args, data, y_pred)
        
        ########### PLOT CLUSTER CORR ####################################################
        
        plot_cluster_corr(cm, save=output_dir)

        ########### PLOT CLUSTER HEATMAP #################################################

        plot_cluster_heatmap(cm, lb, save=output_dir)
        
        ########### DIM. REDUCTION #######################################################
        
        res, pca = reduce_dim(args, cm, lb, plot=True, save=output_dir)

        ########### CHECK FOR BATCH EFFECTS ##############################################

        batch_res = check_batch_effects(args, res, plot=True, save=output_dir)
        
        ########### TRAIN CLASSIFIER ######################################################
        
        model, accuracy = train_classifier(args, res, plot=True, save=output_dir)
        
        ########### APPLY TO UNLABELED CELL LINES #########################################

        prob_res = predict_mutants(args, model, res, batch_res, low_data_flags)
        
        ########### SAVE RESULTS ##########################################################

        _sens = args.sensitive_line[0].upper()
        _res = args.resistant_line[0].upper()
        _drug = args.drug[0].lower()
        
        prob_res.to_csv(output_dir + '/unlabeled_lines_results.csv') 
        
        run_res = pd.DataFrame({'accuracy(train)': accuracy, 
                                'pc1_var':pca.explained_variance_ratio_[0], 
                                'pc2_var':pca.explained_variance_ratio_[1], 
                                'kmeans_inertia': km.inertia_,
                                'res_line':_res, 
                                'sens_line': _sens,
                                'drug':_drug, 
                                'nclus':args.nclus[0],
                                'resample_sz': args.resample_sz[0],
                                'load': args.load[0], 
                                'run_id':run_id}, index=[0])
        
        run_res.to_csv(output_dir + '/run_results.csv')
                         
                         
                         















