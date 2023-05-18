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
Our model achieves the following performance on :

<!---
## Citation
If you find this repository useful, please consider citing the following papers: 
--->
