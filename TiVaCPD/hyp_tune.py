# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import sys
import numpy as np
import argparse
from matplotlib import pyplot as plt
import seaborn as sns
from tvgl import *
from load_data import *
import matplotlib
from performance import *
import roerich
from cpd_methods import *
from performance import *
import fnmatch,os
import pickle as pkl
from scipy.signal import savgol_filter
import warnings
from scipy.signal import peak_prominences
from pyampd.ampd import find_peaks, find_peaks_adaptive
import ruptures as rpt
import itertools, random

def save_data(path, array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

def peak_prominences_(distances):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        all_peak_prom = peak_prominences(distances, range(len(distances)))
    return all_peak_prom

def post_processing(score, threshold):
        score_peaks = peak_prominences_(np.array(score))[0]
        for j in range(len(score_peaks)):
            if peak_prominences_(np.array(score))[0][j] - peak_prominences_(np.array(score))[0][j-1] >threshold :
                score_peaks[j] = 1
            else:
                score_peaks[j] = 0
        return score_peaks

def main():

    # load the data
    if args.data_type in ['simulated_data', 'simulated']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'series_*'))
        X_samples = []
        y_true_samples = []
        for i in range(n_samples):
            X, gt_cor, gt_var, gt_mean = load_simulated(data_path, i)

            if np.all((gt_mean== gt_mean[0])) and np.all((gt_var == gt_var[0])):
                # cases where we have changes only in correlation
                y_true = gt_cor
            else:
                y_true = abs(gt_mean) + abs(gt_var) + abs(gt_cor)
            y_true_spike = y_true.copy()
            for j in range(len(y_true)):
                if y_true[j] != y_true[j-1] and j!=0:
                    y_true_spike[j] = 1
                else:
                    y_true_spike[j] = 0
            y_true = y_true_spike
            X_samples.append(X)
            y_true_samples.append(y_true)

    if args.data_type in ['block_correlation', 'block']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'series_*'))
        X_samples = []
        y_true_samples = []
        for i in range(n_samples):
            X, gt_cor, gt_var, gt_mean = load_simulated(data_path, i)
            y_true = abs(gt_cor[:,0])
            y_true[:]=0
            y_true_spike = y_true.copy()
            for j in range(len(y_true)):
                if j==int((2/3)*100) or j==int((1/3)*100) :
                    y_true_spike[j] = 1
                else:
                    y_true_spike[j] = 0
            
            y_true = y_true_spike
            X_samples.append(X)
            y_true_samples.append(y_true)

    if args.data_type in ['beewaggle', 'beedance']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'*.mat'))
        X_samples = []
        y_true_samples = []
        
        for i in range(1,n_samples):
            X, y_true = load_beedance(data_path, i)
            X_samples.append(X)
            y_true_samples.append(y_true)

    if args.data_type in ['HAR', 'har']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'HAR_X_*'))
        X_samples = []
        y_true_samples = []
        for i in range(n_samples):
            X, y_true = load_HAR(data_path, i)
            y_true_spike = y_true.copy()
            for j in range(len(y_true)):
                if y_true[j] != y_true[j-1]:
                    y_true_spike[j] = 1
                else:
                    y_true_spike[j] = 0
            y_true = y_true_spike
            y_true[0] = 0
            X_samples.append(X)
            y_true_samples.append(y_true)
            
    # results path
    if not os.path.exists(os.path.join(args.out_path)):
        os.mkdir(os.path.join(args.out_path))
    if not os.path.exists(os.path.join(args.out_path, args.exp)):
        os.mkdir(os.path.join(args.out_path, args.exp))
    if not os.path.exists(os.path.join(args.out_path, args.exp)):
        os.mkdir(os.path.join(args.out_path, args.exp, args.model_type))
    

    if args.data_type in ['HAR', 'har']:
        x_train, x_test, y_train, y_test = seg_data(X_samples,y_true_samples, 'HAR')
    else:
        x_train, x_test, y_train, y_test = seg_data(X_samples,y_true_samples, 'others')

    #Hyper-parameters tuning
    #alpha_set = np.linspace(0.2, 1 , 3)     
    #beta_set = np.logspace(0, 1.3, 3) 
    if(args.wavelet):
        hyp_params = {'threshold':[.2,.02,.002],'slice_size':[14, 10, 5],'alpha_':[5, 1, 0.4],'beta':[12, 6, 0.4],
                'wave_shape':['gaus1','mexh','shan','cgau3'],'wave_ext':['symmetric','smooth']}
        best_params = [.2, 10, 0.4, 0.4, 'mexh','smooth']
        comb = [list(np.arange(3)),list(np.arange(3)),list(np.arange(3)),list(np.arange(3)), list(np.arange(4)), list(np.arange(2))]
    else:
        hyp_params = {'threshold':[.2,.02,.002],'slice_size':[14, 10, 5],'alpha_':[5, 1, 0.4],'beta':[12, 6, 0.4]}
        best_params = [.2, 10, 0.4, 0.4]
        comb = [list(np.arange(3)),list(np.arange(3)),list(np.arange(3)),list(np.arange(3))]
    
    best_dev_f1_score = 0
    grid_index = list(itertools.product(*comb))
    random_index = random.choices(grid_index, k=20)
    #params = {key: random.sample(value, 1)[0] for key, value in hyp_params.items()}
    #For bayesian optimization
    #https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
    #https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Kaggle%20Version%20of%20Bayesian%20Hyperparameter%20Optimization%20of%20GBM.ipynb
    f1_list = []
    print('length data for hyp tuning: ', len(x_train))
    for ind in random_index: 
        print('rand indices: ', ind)
        threshold = list(hyp_params['threshold'])[ind[0]]
        slice_size = list(hyp_params['slice_size'])[ind[1]]
        alpha_ = list(hyp_params['alpha_'])[ind[2]]
        beta = list(hyp_params['beta'])[ind[3]]
        if(args.wavelet):
            wave_shape = list(hyp_params['wave_shape'])[ind[4]]
            wave_ext = list(hyp_params['wave_ext'])[ind[5]]
        for i in range(len(x_train)):
            print(i)
            X = x_train[i]
            y_true = y_train[i]
            f1_scores = []
            data_path = os.path.join(args.out_path, args.exp)
            data_path += '_'+str(args.penalty_type)
            model = MMDATVGL_CPD(X, max_iters = args.max_iters, overlap=args.overlap, alpha = 0.001, threshold = threshold, f_wnd_dim = args.f_wnd_dim, p_wnd_dim = args.p_wnd_dim, data_path = data_path, sample = i,
                                slice_size=slice_size, alpha_=alpha_, beta=beta) 
            mmd_score = model.mmd_score #shift(model.mmd_score, args.p_wnd_dim)
            corr_score = model.corr_score
            minLength = min(len(mmd_score), len(corr_score)) 
            corr_score = (corr_score)[:minLength]
            mmd_score = mmd_score[:minLength] 
            y_true = y_true[:minLength]

            mmd_score_nor = stats.zscore(mmd_score)
            corr_score_nor = stats.zscore(corr_score)
            q3, q1 = np.percentile(corr_score_nor, [75 ,25])
            corr_iqr = q3 - q1
            corr_vote = (corr_score_nor < (1.5 * corr_iqr))
            corr_score_nor = corr_score_nor * corr_vote.astype(int)

            mmd_score_savgol  = mmd_score  
            corr_score_savgol = savgol_filter(corr_score, 7, 1)
            combined_score_savgol  = savgol_filter(np.add(abs(mmd_score_savgol), abs(corr_score_savgol)), 11,   1) 

            all_scores = [mmd_score_nor, corr_score_nor, corr_score_savgol, combined_score_savgol]
            all_scores = np.transpose(np.array(all_scores))

            w_corr = True
            if(w_corr):
                W = np.cov(all_scores.T)
                W = np.sum(W , axis = 0)
                print('weight W: ', W)
            else:
                score_num = all_scores.shape[1]
                D = np.zeros((score_num,score_num))
                for i in range(score_num):
                    for j in range(i+1, score_num):
                        D[i,j] =  np.mean(abs(all_scores[:,i] - all_scores[:,j]))
                        D[j,i] = D[i,j]
                W = np.sum(D, axis = 0) 
                print('weight D: ', W)
            final_score = np.dot(all_scores, W) / sum(W)
        
            metrics = ComputeMetrics(y_true, final_score, args.margin)
            f1_score = np.round(metrics.f1,2)
            print('f1 score: ', f1_score)
            f1_scores.append(f1_score)

        f1_scores = np.mean(f1_scores)
        f1_list.append(f1_scores)
        if f1_scores > best_dev_f1_score:
            best_dev_f1_score = f1_scores
            best_params = [threshold, slice_size, alpha_, beta, wave_shape, wave_ext]

    threshold, slice_size, alpha_, beta, wave_shape, wave_ext = best_params
    print('f1 scores: ', f1_list)
    print(mean_confidence_interval(f1_list))
    print('best params: ', best_params)
    print('best f1 score: ', best_dev_f1_score)
    """
    for i in range(0, len(x_test)):
        print(i)
        if args.model_type == 'MMDATVGL_CPD':
            
            data_path = os.path.join(args.out_path, args.exp)

            X = X_samples[i]
            y_true = y_test[i]
            
            model = MMDATVGL_CPD(X, max_iters = args.max_iters, overlap=args.overlap, alpha = 0.001, threshold = threshold, f_wnd_dim = args.f_wnd_dim, p_wnd_dim = args.p_wnd_dim, data_path = data_path, sample = i,
                                 slice_size=slice_size, alpha_=alpha_, beta=beta, wave_shape=wave_shape, wave_ext=wave_ext) 
                
            mmd_score = model.mmd_score #shift(model.mmd_score, args.p_wnd_dim)
            corr_score = model.corr_score
            
            minLength = min(len(mmd_score), len(corr_score)) 
            corr_score = (corr_score)[:minLength]
            mmd_score = mmd_score[:minLength] 
            combined_score = np.add(abs(mmd_score), abs(corr_score)) 
            y_true = y_true[:minLength]

            # processed combined score

            mmd_score_savgol  = mmd_score #savgol_filter(mmd_score, 11, 1)  
            corr_score_savgol = savgol_filter(corr_score, 7, 1) 
            
            #if not np.all((mmd_score_savgol == 0)):
            #    mmd_score_savgol /= np.max(np.abs(mmd_score_savgol),axis=0)
            #if not np.all((corr_score_savgol == 0)):
            #    corr_score_savgol /= np.max(np.abs(corr_score_savgol),axis=0)
            
            combined_score_savgol  = savgol_filter(np.add(abs(mmd_score_savgol), abs(corr_score_savgol)), 11,   1) 

            if not np.all((combined_score_savgol == 0)):
                combined_score_savgol /= np.max(np.abs(combined_score_savgol),axis=0)
            
            # save intermediate results
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path) 
            if not os.path.exists(data_path): 
                os.mkdir(data_path) 

            save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X) 
            save_data(os.path.join(data_path, ''.join(['y_true_', str(i), '.pkl'])), y_true) 
            save_data(os.path.join(data_path, ''.join(['mmd_score_', str(i), '.pkl'])), mmd_score) 
            save_data(os.path.join(data_path, ''.join(['corr_score_', str(i), '.pkl'])), corr_score)

        
            y_pred = mmd_score
            metrics = ComputeMetrics(y_true, y_pred, args.margin)
            print("DistScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            y_pred = corr_score
            metrics = ComputeMetrics(y_true, y_pred, args.margin)
            print("CorrScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            y_pred = combined_score
            metrics= ComputeMetrics(y_true, y_pred, args.margin)
            print("EnsembleScore:", "AUC:", np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            print("Processed:")

            y_pred = mmd_score_savgol
            metrics = ComputeMetrics(y_true, y_pred, args.margin)
            auc_scores_mmdagg.append(metrics.auc)
            f1_scores_mmdagg.append(metrics.f1) 
            precision_scores_mmdagg.append(metrics.precision)
            recall_scores_mmdagg.append(metrics.recall)
            print("DistScore:", "AUC:", np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            y_pred = corr_score_savgol
            metrics = ComputeMetrics(y_true, y_pred, args.margin)
            auc_scores_correlation.append(metrics.auc)
            f1_scores_correlation.append(metrics.f1) 
            precision_scores_correlation.append(metrics.precision)
            recall_scores_correlation.append(metrics.recall)
            print("CorrScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            

            y_pred = combined_score_savgol
            metrics = ComputeMetrics(y_true, y_pred, args.margin)
            auc_scores_combined.append(metrics.auc)
            f1_scores_combined.append(metrics.f1)
            precision_scores_combined.append(metrics.precision)
            recall_scores_combined.append(metrics.recall)
            print("EnsembleScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            peaks=metrics.peaks
            
            plt.figure(figsize= (30,3))
            plt.plot(X)
            plt.plot(y_true, label = 'y_true')
            plt.legend()
            plt.title(args.exp)
            plt.savefig(os.path.join(data_path, ''.join(['X_', str(i), '.png'])))
            plt.clf()

            plt.figure(figsize= (30,3))
            plt.plot(mmd_score_savgol, label = 'mmd_score_savgol')
            plt.plot(corr_score_savgol, label = 'corr_score_savgol')
            plt.plot(combined_score_savgol, label = 'combined_score_savgol')
            plt.plot(y_true, label = 'y_true')
            plt.legend()
            plt.title(args.exp)
            plt.savefig(os.path.join(data_path, ''.join(['TiVaCPD_score_components_', str(i), '.png'])))
            plt.clf()

            plt.figure(figsize= (30,3))
            plt.plot(y_true, label = 'y_true')
            plt.plot(peaks, label = 'peaks')
            plt.legend()
            plt.title(args.exp)
            plt.savefig(os.path.join(data_path, ''.join(['TiVaCPD_score_components_peaks_', str(i), '.png'])))
            plt.clf()

            print(args.data_type, args.model_type, args.exp, args.score_type)
    #"""
    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--data_path',  default='./data/changing_correlation') # exact data dir, including the name of exp
    parser.add_argument('--out_path', default = './out2') # just the main out directory
    parser.add_argument('--data_type', default = 'simulated_data') # others: beedance, HAR, block
    parser.add_argument('--max_iters', type = int, default = 500)
    parser.add_argument('--overlap', type = int, default = 1)
    parser.add_argument('--threshold', type = float, default = .2)
    parser.add_argument('--f_wnd_dim', type = int, default = 10)
    parser.add_argument('--p_wnd_dim', type = int, default = 3)
    parser.add_argument('--exp', default = 'changing_correlation') # used for output path for results
    parser.add_argument('--model_type', default = 'MMDATVGL_CPD')
    parser.add_argument('--score_type', default='combined') # others: combined, correlation, mmdagg
    parser.add_argument('--margin', default = 5)
    parser.add_argument('--slice_size', default = 7)
    parser.add_argument('--ckpt', help='model\'s checkpoint location')
    parser.add_argument('--penalty_type', default = 'L1', help='The TVGL penalty type; corrently accepts L1, L2, and perturbed')
    parser.add_argument('--wavelet', default = False, type= bool)

    args = parser.parse_args()

    main()

        
