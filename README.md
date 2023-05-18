# TiVaCPD: Dynamic Interpretable Change Point Detection
This repository is the official implementation of TiVaCPD change point detection paper. The overview of the methodology is shown below.
For more information on this work please read [Dynamic Interpretable Change Point Detection]().


![Network Overview](https://github.com/Tinbeh97/TiVaCPD/blob/main/Overview.jpg "network overview")
## Requirements
All codes are written for Python 3.9.16 (https://www.python.org/) on Linux platform. The required packages are in the requiements (TiVaCPD/requirements.txt) file.

To install requirements:

```setup
pip install -r requirements.txt
```
<!---
### Clone this repository
```
git clone git@github.com:
```
--->
## Previous work

This represetory contains the implementation for running the following benchmark methods.
- Kernel Change Point Detection with Auxiliary Deep Generative Models (KLCPD) (Chang et al., 2019) [KLCPD](https://arxiv.org/abs/1901.06077)
- Generalization of Change-Point Detection in Time Series Data Based on Direct Density Ratio Estimation (Hushchyn & Ustyuzhanin, 2021) [roerich](https://arxiv.org/abs/2001.06386)
- Estimating dynamic graphical models from multivariate time-series data (GraphTime) (Gibberd & Nelson, 2015) [GRAPHTIME_CPD](https://ceur-ws.org/Vol-1425/paper9.pdf)
- Change Point Detection in Time Series Data using Autoencoders with a Time-Invariant Representation (TIRE) De Ryck et al. (2021) [ruptures](https://arxiv.org/abs/2008.09524)

## Datasets 
The TiVACPD method was tested on 4 simulated datasets and 2 real world datasets. All datasets can be found in the [data](./TiVaCPD/data) folder.
For generating the simulated datasets use the following commands:
- For Jumping Mean, changing_variance, and changing correlation CPs
```
python3 simulate.py --path saving_path --constant_mean True_or_False --constant_var True_or_False --constant_corr True_or_False
```
- For Arbitrary CPs
```
python3 simulate2.py --path saving_path
```

The real-life datasets can be found here:
- Beedance dataset
- Human activity recognition (HAR) dataset [link](https://paperswithcode.com/dataset/har)
## Hyperparameters tuning 
```
python3 hyp_tune.py --data_path ./data/HAR_new --exp 'results folder name' --model_type 'model type' --data_type 'data type' --penalty_type L2
```

For example for the HAR data and the proposed TiVaCPD model the command would be like:
```
python3 hyp_tune.py --data_path ./data/HAR_new --exp HAR --model_type MMDATVGL_CPD --data_type HAR --penalty_type L2
```
## Training and Inference
```
python3 experiments.py --data_path 'path_to_data' --exp 'name for saving the results' --model_type 'which_model_do_you_like_to_test' --data_type 'dataset type' --penalty_type L2 --ensemble_win True
```

For example for the HAR data and the proposed TiVaCPD model the command would be as following:
```
python3 experiments.py --data_path ./data/HAR_new --exp HAR --model_type MMDATVGL_CPD --data_type HAR --penalty_type L2 --ensemble_win True
```
## Saved Models and results
The saved models' results can be found in the [out2](./TiVaCPD/out2) folder based on output file name 'exp' that was given to them during experiments and the tvgl's penalty type that was employed.

## Results
Our model achieves the following performance on HAR and Bee Dance datasets:
|-------------------------------------|-------------|
| Bee Dance      |             |             | HAR     |             |             |
| Method                              | Precision   | Recall      | F1(M=5)              | Precision    | Recall      | F1(M=5)              |
| \ourCPD                             | 0.36 (0.18) | 0.59 (0.22) | \textbf{0.45 (0.15)} | 0.72 (0.06)  | 0.48 (0.06) | \textbf{0.58 (0.06)} |
| \ourCPD-CovScore                    | 0.34 (0.26) | 0.36 (0.19) | 0.34 (0.21)          | 0.62 (0.06)  | 0.56 (0.05) | 0.57 (0.04)          |
| \ourCPD-DistScore                   | 0.48 (0.11) | 0.25 (0.13) | 0.32 (0.11)          | 0.50 (0.06)  | 0.35 (0.03) | 0.40 (0.03)          |
| KL-CPD                              | 0.24 (0.26) | 0.10 (0.07) | 0.13 (0.11)          | 0.66 (0.11)  | 0.20 (0.03) | 0.30 (0.04)          |
| Roerich                             | 0.50 (0.34) | 0.32 (0.26) | 0.40 (0.30)          | 0.69 (0.15)  | 0.11 (0.03) | 0.18 (0.05)          |
| GraphTime                           | 0.13 (0.04) | 0.77 (0.13) | 0.22 (0.07)          | 0.04 (0.002) | 0.96 (0.02) | 0.08 (0.01)          |
| TIRE                                | 0.34 (0.44) | 0.14 (0.19) | 0.20 (0.26)          | 0.52 (0.19)  | 0.14 (0.05) | 0.22 (0.08)          |


<!---
## Citation
If you find this repository useful, please consider citing the following papers: 
--->
