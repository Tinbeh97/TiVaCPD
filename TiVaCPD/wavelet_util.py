import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt

def wavelet_t_win(data, feat_id=None, width_num = 5, wavelet='mexh', mode='smooth', ext_len=5, plot_wav=False):
    if(feat_id==None):
        feat_id = np.arange(data.shape[1])
    #feat_vec = data
    feat_vec = np.empty(data.shape, data.dtype)
    for id in list(feat_id):
        par_data = data[:,id]
        widths = np.arange(1, width_num)
        if(mode==None):
            cwtmatr, freqs = pywt.cwt(par_data, widths, wavelet) #mexh
            transform_data = np.array(cwtmatr)
        else:
            ext_data = pywt.pad(par_data, ext_len, mode)
            cwtmatr, freqs = pywt.cwt(ext_data, widths, wavelet) #mexh
            transform_data = np.array(cwtmatr)[:, ext_len:-ext_len]
        feat_vec = np.append(feat_vec, transform_data.T, axis=1)

        if(plot_wav):
            x = np.arrange(len(par_data))
            cwtmatr2, freqs = pywt.cwt(par_data, widths, wavelet) #mexh
            # freq: detailed_coef
            for i in range(len(widths)):
                plt.figure()
                plt.plot(x, par_data,'b')
                plt.plot(x, cwtmatr2[i],'-r')
                plt.plot(x, transform_data[i],'--g')
                plt.show()

    return np.array(feat_vec) #+ np.random.normal(.1, .5, feat_vec.shape)#np.finfo(np.float32).eps
        
def plot_wav(name='fbsp', scale=5):
    wav = pywt.ContinuousWavelet(name)
    width = wav.upper_bound - wav.lower_bound
    max_len = int(3*width + 1)
    t = np.arange(max_len)
    int_psi, x = pywt.integrate_wavelet(wav, precision=10)
    # The following code is adapted from the internals of cwt
    #int_psi, x = pywt.integrate_wavelet(wav, precision=10)
    step = x[1] - x[0]
    j = np.floor(
        np.arange(scale * width + 1) / (scale * step))
    if np.max(j) >= np.size(int_psi):
        j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
    j = j.astype(np.int_)

    # normalize int_psi for easier plotting
    int_psi /= np.abs(int_psi).max()

    # discrete samples of the integrated wavelet
    filt = int_psi[j][::-1]

    # The CWT consists of convolution of filt with the signal at this scale
    # Here we plot this discrete convolution kernel at each scale.

    nt = len(filt)
    t = np.linspace(-nt//2, nt//2, nt)
    plt.plot(t, filt.real, t, filt.imag)