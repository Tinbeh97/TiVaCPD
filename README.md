# TiVaCPD: Dynamic Interpretable Change Point Detection
This repository is the official implementation of TiVaCPD change point detection paper. The overview of the methodology is shown below.
For more information on this work please read [Dynamic Interpretable Change Point Detection]().


![Network Overview](https://github.com/Tinbeh97/TiVaCPD/blob/main/Overview.jpg "network overview")
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Prerequisites

All codes are written for Python 3.9.16 (https://www.python.org/) on Linux platform. The required packages are in the requiements (TiVaCPD/requirements.txt) file.

%### Clone this repository
%```
%git clone git@github.com:
%```
## Previous work

This represetory contains the codes for previous work.

## Datasets 
6 dataset - 3 simulated and 3 real world data 

Occupancy detection dataset {link}(https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+#)
simulate.py is for simulating data -> Jumping Mean, changing_variance, changing correlation
simulate2.py -> Arbitrary CPs

## Training 

```
python3 experiments.py --data_path ./data/beedance --exp beedance --model_type MMDATVGL_CPD --data_type beedance --penalty_type L2 --ensemble_win True
```
## Hyperparameters tuning 

```
python3 hyp_tune.py --data_path ./data/beedance --exp beedance --model_type MMDATVGL_CPD --data_type beedance --penalty_type L2
```
## Inference

## Pre-trained Models

%## Citation
%If you find this repository useful, please consider citing the following papers: 
