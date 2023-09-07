import numpy as np
import random 
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import os

def generate_dataset(period=200, N=2000, add_noise=False, mu_change=True):
    mu = 0
    sigma = 1.
    N = 1

    T = [0, 1]
    X = [np.random.normal(mu, sigma, 1)[0], np.random.normal(mu, sigma, 1)[0]]
    cps = []

    for i in range(2, N):
        if i % period == 0:
            N += 1
            if(mu_change):
                mu += 0.5 * N
            else:
                sigma += 0.25 * N
            cps.append(i)
        T += [i]
        ax = 0.6 * X[i-1] - 0.5 * X[i-2] + np.random.normal(mu, sigma, 1)[0]
        X += [ax]
    
    X = np.array(X).reshape(-1, 1)

    if(add_noise):
        X2 = np.random.normal(0, 1, N).reshape(-1, 1)
        X = X + X2
    return X

def generate_sin_dataset(n_cp=3):
    lengths = np.random.randint(50,100,size=(n_cp,))
    N = np.sum(lengths)
    label = np.zeros((N,))

    mu = 0.5
    sigma = 1.
    w = 1
    X = np.zeros((N,3))
    sigma = np.random.normal(mu, sigma, N*3).reshape(-1, 3)
    total_length = 0
    for i in range(n_cp):
        period = lengths[i]
        time = np.linspace(0, period, period) + total_length
        for sig in range(3):
            # w = 2 * np.pi * f
            periodic_X = np.sin(w * time)
            w = w * np.log(np.exp(1) + 0.5*(i+2))
            X[total_length:total_length+period,sig] = periodic_X
        if (i>0):
            label[total_length-1] = 1
        total_length += period
    X = X + sigma
    return X, label

def sim_data(n_cp=3, constant_mean = False, constant_var = False, constant_corr = False):
    lengths = np.random.randint(50,100,size=(n_cp,))
    N = np.sum(lengths)
    label = np.zeros((N,))
    mu_arr = np.random.randint(0,7,n_cp)
    sigma_arr = np.random.randint(1,10,n_cp)
    corr_arr = np.random.randint(0,3,n_cp)/2
    mu, sigma, rho = 0, 1, 0
    total_length = 0
    for i in range(n_cp):
        X_len = lengths[i]

        if not constant_mean:
            mu = mu_arr[i]
        if not constant_var:
            sigma = sigma_arr[i]
        if not constant_corr:
            rho = corr_arr[i]
        
        mean = [mu]*3
        var = [sigma]*3
        cov_matrix = np.diag(var)
        cov_matrix[1,2] = rho
        cov_matrix[2,1] = rho
        X = np.random.multivariate_normal(mean, cov_matrix, X_len)
        if(i==0):
            series = X.copy()
        else:
            series = np.append(series, X, axis=0)

        #print('mean & corr matrix: ', mean, cov_matrix)
        if(i>0):
            if((mu1 != mu) or (sigma1 != sigma) or (rho1 != rho)):
                label[total_length-1] = 1

        mu1, sigma1, rho1 = mu, sigma, rho
        total_length += X_len
    series = np.array(series)
    return series, label

def sim_data2(n_cp=3, constant_mean = False, constant_var = False, constant_corr = False):
    lengths = np.random.randint(50,100,size=(n_cp,))
    N = np.sum(lengths)
    label = np.zeros((N,))
    mu_arr = np.random.randint(0,7, n_cp*3).reshape(n_cp, 3)
    sigma_arr = np.random.randint(1,10, n_cp*3).reshape(n_cp, 3)
    sigma_arr = [.7,.1,.7,.1,.7]*3 #,.1,.8,.5,.3,.5,.3,.1,.3,.5,.3,.5,.3, .1,.8,.1,.8,.1,.8,.1,.8,.5,.3,.5,.3]
    sigma_arr = [.7,.1,.8,.2,.9]*3
    sigma_arr = np.array(sigma_arr).reshape(3, n_cp).T
    #corr_arr = np.random.randint(0,3, n_cp*2)/2
    #corr_arr = np.random.randint(0,3, n_cp*2) - 1
    #corr_arr = corr_arr.reshape(n_cp, 2)
    corr_arr = np.array([[0,1,0,1,0], [0,0,-1,0,0]]).T
    mu, sigma, rho = [0]*3, [1,1,1], [0,0]
    total_length = 0
    for i in range(n_cp):
        X_len = lengths[i]
        if not constant_mean:
            mu = mu_arr[i]
        if not constant_var:
            sigma = sigma_arr[i]
        if not constant_corr:
            rho = corr_arr[i]

        cov_matrix = np.diag(sigma).astype('float16')
        cov_matrix[1,2] = rho[1]
        cov_matrix[2,1] = rho[1]
        cov_matrix[1,0] = rho[0]
        cov_matrix[0,1] = rho[0]
        
        if False:
            X = np.random.normal(mu[0], sigma[0], X_len).reshape(-1, 1)
            X = np.append(X, np.random.normal(mu[1], sigma[1], X_len).reshape(-1, 1), axis=1)
            X = np.append(X, np.random.normal(mu[2], sigma[2], X_len).reshape(-1, 1), axis=1)
        else: 
            X = np.random.multivariate_normal(mu, cov_matrix, X_len)
        if(i==0):
            series = X.copy()
        else:
            series = np.append(series, X, axis=0)

        print('mean & corr matrix: ', mu, cov_matrix)
        change_Flag = False
        if(i>0):
            for index in range(len(mu)):
                if((mu1[index] != mu[index]) or (sigma1[index] != sigma[index])):
                    change_Flag = True
                    break
            for index in range(len(rho)):
                if(rho1[index] != rho[index]):
                    change_Flag = True
            if(change_Flag):
                label[total_length-1] = 1
            print(change_Flag)

        mu1, sigma1, rho1 = mu, sigma, rho
        total_length += X_len
    series = np.array(series)
    return series, label

def save_data(path, array):
    with open(path,'wb') as f:
        pkl.dump(array, f)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='change point detection')
    parser.add_argument('--path', type=str, default='./data')
    parser.add_argument('--constant_mean', default = 'False')
    parser.add_argument('--constant_var', default = 'False')
    parser.add_argument('--constant_corr', default = 'False')
    parser.add_argument('--exp', default = '0')
    parser.add_argument('--num_cp', type = int, default = 3)
    parser.add_argument('--n_samples', type = int, default = 20)
    parser.add_argument('--periodic_sim', default=False, type=bool)

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.path, args.exp)): 
        os.mkdir(os.path.join(args.path, args.exp))
    data_path = os.path.join(args.path, args.exp)

    for i in range(args.n_samples):
        if(args.periodic_sim):
            X, label = generate_sin_dataset(n_cp = args.num_cp)
        else:
            X, label = sim_data2(n_cp = args.num_cp, constant_mean = eval(args.constant_mean), 
                                                        constant_var = eval(args.constant_var), 
                                                        constant_corr = eval(args.constant_corr))
        """
        print('shape of X: ', X.shape)
        time = np.arange(len(X))
        fig, axs = plt.subplots(3, 1)
        for j in range(X.shape[1]):
            axs[j].plot(time, X[:,j])
            axs[j].set_xlabel('Time')
            axs[j].set_ylabel('series '+str(j+1))
        loc = np.where(label==1)[0]
        print(loc)
        for xi in loc:
            for j in range(X.shape[1]):
                axs[j].axvline(x = xi, color = 'black', label = 'change point')
        plt.show()
        exit()
        #"""
        data_path = os.path.join(args.path, args.exp)

        save_data(os.path.join(data_path, ''.join(['series_', str(i), '.pkl'])), X)
        save_data(os.path.join(data_path, ''.join(['label_', str(i), '.pkl'])), label)
