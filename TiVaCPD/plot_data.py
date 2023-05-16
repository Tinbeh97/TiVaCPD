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

    x_samples = X_samples[0]
    for i in range(len(x_samples[1])):
        x = x_samples[:,i]
        plt.figure()
        plt.plot(x)
        plt.savefig('image_out/x_'+str(i)+'.png')
        


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

        
