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
from roerich.change_point import ChangePointDetectionClassifier

from cpd_methods import *
from performance import *
import fnmatch,os
import pickle as pkl
from scipy.signal import savgol_filter
import warnings
from scipy.signal import peak_prominences
import scipy.stats as stats
from pyampd.ampd import find_peaks, find_peaks_adaptive
import ruptures as rpt

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

            y_true = np.zeros_like(gt_cor)
            for j in range(len(gt_cor)):
                if gt_cor[j] != gt_cor[j-1] and j!=0:
                    y_true[j] += 1
            for j in range(len(gt_mean)):
                if gt_mean[j] != gt_mean[j-1] and j!=0:
                    y_true[j] += 1
            for j in range(len(gt_var)):
                if gt_var[j] != gt_var[j-1] and j!=0:
                    y_true[j] += 1
            y_true[y_true >= 1] = 1

            X_samples.append(X)
            y_true_samples.append(y_true)

    if args.data_type in ['new_simulated']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'series_*'))
        X_samples = []
        y_true_samples = []
        for i in range(n_samples):
            X, labels = load_simulated_new(data_path, i)
            X_samples.append(X)
            y_true_samples.append(labels)

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
        
        for i in range(1,n_samples+1):
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

    if args.data_type in ['OCCUPANCY', 'occupancy']:
        data_path = os.path.join(args.data_path)
        n_samples = len(fnmatch.filter(os.listdir(data_path),'occupancy_X_*'))
        X_samples = []
        y_true_samples = []
        for i in range(n_samples):
            X, y_true = load_occupancy(data_path, i)
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
    
    auc_scores_combined =  []
    f1_scores_combined = []
    precision_scores_combined = []
    recall_scores_combined = []
    auc_scores_correlation =  []
    f1_scores_correlation= []
    precision_scores_correlation = []
    recall_scores_correlation = []
    auc_scores_mmdagg =  []
    f1_scores_mmdagg= []
    precision_scores_mmdagg = []
    recall_scores_mmdagg = []
    final_auc_scores_combined =  []
    final_f1_scores_combined = []
    final_precision_scores_combined = []
    final_recall_scores_combined = []

    auc_scores = []
    f1_scores = []
    precision_scores  =[]
    recall_scores = []
    plot_verbose = False

    for i in range(0, len(X_samples)):
        print('sample: ', i)
        if args.model_type == 'MMDATVGL_CPD':
            
            data_path = os.path.join(args.out_path, args.exp)
            data_path += '_'+str(args.penalty_type)
            if(args.wavelet):
                data_path += '_wavelet'

            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path) 
            if not os.path.exists(data_path): 
                os.mkdir(data_path) 

            X = X_samples[i]
            y_true = y_true_samples[i]


            model = MMDATVGL_CPD(X, max_iters = args.max_iters, overlap=args.overlap, alpha = 0.001, threshold = args.threshold, f_wnd_dim = args.f_wnd_dim, p_wnd_dim = args.p_wnd_dim, 
                                 data_path = data_path, sample = i, slice_size=args.slice_size,  penalty_type = args.penalty_type, wavelet = False, data_type=args.data_type, plot_verbose=plot_verbose) 

            mmd_score = model.mmd_score #shift(model.mmd_score, args.p_wnd_dim)
            corr_score = model.corr_score
            if(plot_verbose):
                plt.figure()
                plt.plot(mmd_score)
                plt.savefig('image_out/mmd_score'+'.png')
                plt.clf()
                plt.figure()
                plt.plot(corr_score)
                plt.savefig('image_out/corr_score'+'.png')
                plt.clf()
                plt.close()

            if(args.wavelet):
                model_wav = MMDATVGL_CPD(X, max_iters = args.max_iters, overlap=args.overlap, alpha = 0.001, threshold = args.threshold, f_wnd_dim = args.f_wnd_dim, p_wnd_dim = args.p_wnd_dim, 
                                 data_path = data_path, sample = i, slice_size=args.slice_size,  penalty_type = args.penalty_type, wavelet = True) 
                mmd_score_wave = model_wav.mmd_score
                corr_score_wave = model_wav.corr_score

            print('offset value: ', model.offset)
            
            minLength = min(len(mmd_score), len(corr_score)) 
            if(args.wavelet):
                minLength_wav = min(len(mmd_score_wave), len(corr_score_wave))
                print('min lengths: ',len(mmd_score_wave), len(corr_score_wave), minLength_wav, minLength)
                minLength = min(minLength, minLength_wav)
                mmd_score_wave = mmd_score_wave[:minLength] 
                corr_score_wave = corr_score_wave[:minLength] 

            corr_score = (corr_score)[:minLength]
            mmd_score = mmd_score[:minLength] 
            combined_score = np.add(abs(mmd_score), abs(corr_score)) 
            y_true = y_true[:minLength]


            # processed combined score

            mmd_score_savgol  = savgol_filter(mmd_score, 7, 1)  
            mmd_score_savgol = np.maximum (0, mmd_score_savgol)
            corr_score_savgol = savgol_filter(corr_score, 7, 1) #savgol_filter(x, window_length, polyorder
            corr_score_savgol = np.maximum (0, corr_score_savgol)
            #if not np.all((mmd_score_savgol == 0)):
            #    mmd_score_savgol /= np.max(np.abs(mmd_score_savgol),axis=0)
            #if not np.all((corr_score_savgol == 0)):
            #    corr_score_savgol /= np.max(np.abs(corr_score_savgol),axis=0)
            
            combined_score_savgol  = savgol_filter(np.add(mmd_score, corr_score_savgol), 11,   1) 
            combined_score_savgol = np.maximum (0, combined_score_savgol)

            if not np.all((combined_score_savgol == 0)):
                combined_score_savgol /= np.max(combined_score_savgol,axis=0) 
            #"""
            mmd_score_nor = stats.zscore(mmd_score, axis=0)
            corr_score_nor = stats.zscore(corr_score, axis=0)
            do_ensemble = True
            if(np.sum(np.isnan(mmd_score_nor))>=1 or np.sum(np.isnan(corr_score_nor))>=1):
                do_ensemble = False
                print('not doing ensemble')
            #"""
            select_score, score_final = compare_scores(corr_score, mmd_score, corr_score_savgol, mmd_score_savgol, args.margin, args.score_threshold)
            q3, q1 = np.percentile(corr_score_nor, [75 ,25])
            corr_iqr = q3 - q1
            corr_vote = (corr_score_nor < (1.5 * corr_iqr))
            corr_score_nor = corr_score_nor * corr_vote.astype(int)
            #"""
            w_corr = False
            if(not(do_ensemble) or select_score):
            #if((model.tvgl_iter> 200) & (model.offset> 0.01)):
                if(select_score):
                    y_pred = score_final
                else: 
                    y_pred = combined_score_savgol
            else:
                all_scores = [mmd_score_nor, corr_score_nor, corr_score_savgol, combined_score_savgol]
                print('all score shape: ', np.array(all_scores).shape)

                if(args.wavelet):
                    mmd_score_wave_nor = stats.zscore(mmd_score_wave, axis=0)
                    """
                    corr_score_wave_nor = stats.zscore(corr_score_wave, axis=1)
                    q3, q1 = np.percentile(corr_score_wave_nor, [75 ,25])
                    corr_iqr = q3 - q1
                    corr_vote = (corr_score_wave_nor < (1.5 * corr_iqr))
                    corr_score_wave_nor = corr_score_wave_nor * corr_vote.astype(int)
                    #"""
                    all_scores.append(mmd_score_wave_nor)
                    #all_scores.append(corr_score_wave_nor)
                all_scores = np.transpose(np.array(all_scores))
                print('all score shape: ', all_scores.shape)

                if(args.ensemble_win):
                    final_score = windowed_ensemble(all_scores, window_size=21, w_corr=w_corr, plot_w=plot_verbose)
                    print(np.array(mmd_score_nor).shape, final_score.shape)
                else:
                    if(w_corr):
                        W = np.cov(all_scores.T)
                        W = np.sum(W , axis = 0)
                        print('weight W: ', W)
                        final_score = final_score / sum(W)
                        #final_score = np.dot(all_scores, W) / sum(W)
                        y_pred = final_score
                        metrics= ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
                        print("W Weighted ensemble score:", "AUC:", np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
                    else:
                        score_num = all_scores.shape[1]
                        D = np.zeros((score_num,score_num))
                        for i in range(score_num):
                            for j in range(i+1, score_num):
                                #D[i,j] =  np.mean(abs(all_scores[:,i] - all_scores[:,j]))
                                D[i,j] =  np.mean(abs(all_scores[:,i] - np.mean(all_scores[:,j])))
                                D[j,i] = D[i,j]
                        if(plot_verbose):
                            plt.figure()
                            plt.imshow(D)
                            plt.colorbar()
                            plt.axis('off')
                            plt.savefig('image_out/w_all'+'.png')
                        W = np.sum(D, axis = 0) 
                        print('weight D: ', W)
                        final_score = np.dot(all_scores, W) / sum(W)
                #"""
                #"""
                if(np.isnan(final_score).any()):
                    final_score = np.nan_to_num(final_score)
                    print('There was nan in final score')
                y_pred = final_score
            metrics= ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            name_w = 'W ' if w_corr else 'D '
            print(name_w + "Weighted ensemble score:", "AUC:", np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            final_auc_scores_combined.append(np.round(metrics.auc,3))
            final_f1_scores_combined.append(np.round(metrics.f1,3))
            final_precision_scores_combined.append(np.round(metrics.precision,3))
            final_recall_scores_combined.append(np.round(metrics.recall,3))
            y_pred = mmd_score_nor
            mmd_score_nor = y_pred
            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = 
                                     args.score_threshold)
            print("Norm DistScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            y_pred = corr_score_nor
            corr_score_nor = y_pred
            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            print("Norm CorrScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            combined_score_nor = np.add(mmd_score_nor, corr_score_nor)
            y_pred = combined_score_nor
            metrics= ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            print("Norm EnsembleScore:", "AUC:", np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            #"""

            save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X) 
            save_data(os.path.join(data_path, ''.join(['y_true_', str(i), '.pkl'])), y_true) 
            save_data(os.path.join(data_path, ''.join(['mmd_score_', str(i), '.pkl'])), mmd_score) 
            save_data(os.path.join(data_path, ''.join(['corr_score_', str(i), '.pkl'])), corr_score)

        
            y_pred = mmd_score_savgol
            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            print("DistScore savgol:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            y_pred = corr_score
            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            print("CorrScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            y_pred = combined_score
            metrics= ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            print("EnsembleScore:", "AUC:", np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            print("Processed:")

            y_pred = mmd_score
            """
            print('mmd min: ', np.min(y_true), np.min(y_pred))
            print('mmd max: ', np.max(y_true), np.max(y_pred))
            print('mmd mean: ', np.mean(y_true), np.mean(y_pred))
            #"""
            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            auc_scores_mmdagg.append(metrics.auc)
            f1_scores_mmdagg.append(metrics.f1) 
            precision_scores_mmdagg.append(metrics.precision)
            recall_scores_mmdagg.append(metrics.recall)
            print("DistScore:", "AUC:", np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))

            y_pred = corr_score_savgol
            """
            print('corr min: ', np.min(y_true), np.min(y_pred))
            print('corr max: ', np.max(y_true), np.max(y_pred))
            print('corr mean: ', np.mean(y_true), np.mean(y_pred))
            #"""
            metrics = ComputeMetrics(y_true, y_pred, args.margin, args.score_threshold)
            auc_scores_correlation.append(metrics.auc)
            f1_scores_correlation.append(metrics.f1) 
            precision_scores_correlation.append(metrics.precision)
            recall_scores_correlation.append(metrics.recall)
            print("CorrScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            

            combined_score_savgol  = savgol_filter(combined_score_savgol, 11,   1) 
            combined_score_savgol = np.maximum (0, combined_score_savgol)
            y_pred = combined_score_savgol

            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            auc_scores_combined.append(metrics.auc)
            f1_scores_combined.append(metrics.f1)
            precision_scores_combined.append(metrics.precision)
            recall_scores_combined.append(metrics.recall)
            print("EnsembleScore:", "AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            peaks=metrics.peaks

            if(plot_verbose):
                plt.figure(figsize= (30,3))
                plt.plot(X)
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['X_', str(i), '.png'])))
                plt.clf()
                plt.close()

                plt.figure(figsize= (30,3))
                plt.plot(mmd_score_savgol, label = 'mmd_score_savgol')
                plt.plot(corr_score_savgol, label = 'corr_score_savgol')
                plt.plot(combined_score_savgol, label = 'combined_score_savgol')
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['TiVaCPD_score_components_', str(i), '.png'])))
                plt.clf()
                plt.close()

                plt.figure(figsize= (30,3))
                plt.plot(y_true, label = 'y_true')
                plt.plot(peaks, label = 'peaks')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['TiVaCPD_score_components_peaks_', str(i), '.png'])))
                plt.clf()

                plt.close()

            print('sample: ', i, ' end')
            print(args.data_type, args.model_type, args.exp, args.score_type)
            #exit()
        elif args.model_type == 'GRAPHTIME_CPD':
            X = X_samples[i]
            y_true = y_true_samples[i]
            model = GRAPHTIME_CPD(series = X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, max_iter = 500)
            y_pred = np.zeros((len(y_true)))
            for j in range(len(y_true)):
                if j in model.cps:
                    y_pred[j] = 1 
                else:
                    y_pred[j] = 0
            
            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold, process=False)
            auc_scores.append(metrics.auc)
            f1_scores.append(metrics.f1) 
            precision_scores.append(metrics.precision)
            recall_scores.append(metrics.recall)
            peaks = metrics.peaks
            print("AUC:",np.round(metrics.auc, 2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            if(plot_verbose):
                plt.figure(figsize= (30,3))
                plt.plot(X)
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['X_', str(i), '.png'])))
                plt.clf()
                plt.close()

                plt.figure(figsize= (30,3))
                plt.plot(y_pred, label = 'graphtime')
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['GRAPHTIME_score_', str(i), '.png'])))
                plt.clf()
                plt.close()

                plt.figure(figsize= (30,3))
                plt.plot(y_true, label = 'y_true')
                plt.plot(peaks, label = 'peaks')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['GRAPHTIME_peaks_', str(i), '.png'])))
                plt.clf()
                plt.close()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['graphtime_score_', str(i), '.pkl'])), y_pred)
            

        elif args.model_type == 'KLCPD':
            X = X_samples[i]
            y_true = y_true_samples[i]
            model = KLCPD(X, p_wnd_dim=args.p_wnd_dim, f_wnd_dim=args.f_wnd_dim, epochs=20)
            y_pred = model.scores

            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold, model_type='KLCPD')
            auc_scores.append(metrics.auc)
            f1_scores.append(metrics.f1)
            precision_scores.append(metrics.precision)
            recall_scores.append(metrics.recall)
            peaks=metrics.peaks

            print("AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            if(plot_verbose):
                plt.figure(figsize= (30,3))
                plt.plot(X)
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['X_', str(i), '.png'])))
                plt.clf()

                plt.figure(figsize= (30,3))
                plt.plot(y_pred, label = 'KLCPD')
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['KLCPD_score_', str(i), '.png'])))
                plt.clf()
                
                plt.figure(figsize= (30,3))
                plt.plot(y_true, label = 'y_true')
                plt.plot(peaks, label = 'peaks')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['KLCPD_peaks_', str(i), '.png'])))
                plt.clf()
                plt.close()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['klcpd_score_', str(i), '.pkl'])), y_pred)

        elif args.model_type == 'roerich':
            X = X_samples[i]
            y_true = y_true_samples[i]
            #ChangePointDetectionClassifier
            """
            model = roerich.OnlineNNClassifier(net='default', scaler="default", metric="KL_sym",
                  periods=1, window_size=10, lag_size=30, step=10, n_epochs=25,
                  lr=0.01, lam=0.0001, optimizer="Adam"
                 )
            #"""
            model = ChangePointDetectionClassifier(base_classifier='mlp', metric='klsym', periods=1,
                                     window_size=10, step=1, n_runs=1)
            #score, cps_pred = model.predict(X)

            y_pred, _ = model.predict(X)
            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            auc_scores.append(metrics.auc)
            f1_scores.append(metrics.f1) 
            precision_scores.append(metrics.precision)
            recall_scores.append(metrics.recall)
            peaks =metrics.peaks
            print("AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            if(plot_verbose):
                plt.figure(figsize= (30,3))
                plt.plot(X)
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['X_', str(i), '.png'])))
                plt.clf()

                plt.figure(figsize= (30,3))
                plt.plot(y_pred, label = 'roerich')
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['roerich_score_', str(i), '.png'])))
                plt.clf()

                plt.figure(figsize= (30,3))
                plt.plot(y_true, label = 'y_true')
                plt.plot(peaks, label = 'peaks')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['roerich_peaks_', str(i), '.png'])))
                plt.clf()
                plt.close()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['roerich_score_', str(i), '.pkl'])), y_pred)

        elif args.model_type == 'ruptures':
            X = X_samples[i]

            n_samples = X.shape[0]
            algo = rpt.Pelt(model="rbf").fit(X)
            result = algo.predict(pen=10)

            y_pred=np.zeros(X.shape[0]+1)
            y_pred[result] = 1


            y_true = y_true_samples[i]

            metrics = ComputeMetrics(y_true, y_pred, args.margin, threshold = args.score_threshold)
            auc_scores.append(metrics.auc)
            f1_scores.append(metrics.f1) 
            precision_scores.append(metrics.precision)
            recall_scores.append(metrics.recall)
            peaks = metrics.peaks
            print("AUC:",np.round(metrics.auc,2), "F1:",np.round(metrics.f1,2), "Precision:", np.round(metrics.precision,2), "Recall:",np.round(metrics.recall,2))
            if(plot_verbose):
                plt.figure(figsize= (30,3))
                plt.plot(X)
                plt.plot(y_true, label = 'y_true')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['X_', str(i), '.png'])))
                plt.clf()

                plt.figure(figsize= (30,3))
                plt.plot(y_true, label = 'y_true')
                plt.plot(y_pred, label = 'y_pred')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['ruptures_score_', str(i), '.png'])))
                plt.clf()

                plt.figure(figsize= (30,3))
                plt.plot(y_true, label = 'y_true')
                plt.plot(peaks, label = 'peaks')
                plt.legend()
                plt.title(args.exp)
                plt.savefig(os.path.join(data_path, ''.join(['ruptures_peaks_', str(i), '.png'])))
                plt.clf()
                plt.close()

            data_path = os.path.join(args.out_path, args.exp)
            if not os.path.exists(args.out_path): 
                os.mkdir(args.out_path)
            if not os.path.exists(data_path): 
                os.mkdir(data_path)

            save_data(os.path.join(data_path, ''.join(['ruptures_score_', str(i), '.pkl'])), y_pred)

    

    if args.model_type == 'MMDATVGL_CPD':
        print(f'here: {f1_scores_combined}, {f1_scores_mmdagg}, {f1_scores_correlation}')
        print("auc_scores_combined_CI", mean_confidence_interval(auc_scores_combined))
        print("f1_scores_combined_CI", mean_confidence_interval(f1_scores_combined))
        print("precision_scores_combined_CI", mean_confidence_interval(precision_scores_combined))
        print("recall_scores_combined_CI", mean_confidence_interval(recall_scores_combined))

        print("auc_scores_correlation_CI", mean_confidence_interval(auc_scores_correlation))
        print("f1_scores_correlation_CI", mean_confidence_interval(f1_scores_correlation))
        print("precision_scores_correlation_CI", mean_confidence_interval(precision_scores_correlation))
        print("recall_scores_correlation_CI", mean_confidence_interval(recall_scores_correlation))

        print("auc_scores_mmdagg_CI", mean_confidence_interval(auc_scores_mmdagg))
        print("f1_scores_mmdagg_CI", mean_confidence_interval(f1_scores_mmdagg))
        print("precision_scores_mmdagg_CI", mean_confidence_interval(precision_scores_mmdagg))
        print("recall_scores_mmdagg_CI", mean_confidence_interval(recall_scores_mmdagg))

        print("auc_scores", mean_confidence_interval(final_auc_scores_combined))
        print("f1_scores", mean_confidence_interval(final_f1_scores_combined))
        print("precision_scores", mean_confidence_interval(final_precision_scores_combined))
        print("recall_scores", mean_confidence_interval(final_recall_scores_combined))
        
    else:
        print(args.data_type, args.model_type, args.exp)
        print("auc_CI", mean_confidence_interval(auc_scores))
        print("f1_CI", mean_confidence_interval(f1_scores))
        print("precision_CI", mean_confidence_interval(precision_scores))
        print("recall_CI", mean_confidence_interval(recall_scores))


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--data_path',  default='./data/changing_correlation') # exact data dir, including the name of exp
    parser.add_argument('--out_path', default = './out2') # just the main out directory
    parser.add_argument('--data_type', default = 'simulated_data') # others: beedance, HAR, block
    parser.add_argument('--max_iters', type = int, default = 500)
    parser.add_argument('--overlap', type = int, default = 1)
    parser.add_argument('--threshold', type = float, default = .2)
    parser.add_argument('--score_threshold', type = float, default = .05)
    parser.add_argument('--f_wnd_dim', type = int, default = 10)
    parser.add_argument('--p_wnd_dim', type = int, default = 3)
    parser.add_argument('--exp', default = 'changing_correlation') # used for output path for results
    parser.add_argument('--model_type', default = 'MMDATVGL_CPD')
    parser.add_argument('--score_type', default='combined') # others: combined, correlation, mmdagg
    parser.add_argument('--margin', default = 5)
    parser.add_argument('--slice_size', default = 7)
    parser.add_argument('--wavelet', default = False, type= bool)
    parser.add_argument('--penalty_type', default = 'L1', help='The TVGL penalty type; corrently accepts L1, L2, and perturbed')
    parser.add_argument('--ensemble_win', default = False, help='Whether to find ensemble weights for windowed scores')

    args = parser.parse_args()

    main()

        
