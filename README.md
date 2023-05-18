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

This represetory contains the codes for previous work.

## Datasets 
6 dataset - 3 simulated and 3 real world data 

Occupancy detection dataset {link}(https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+#)
simulate.py is for simulating data -> Jumping Mean, changing_variance, changing correlation
simulate2.py -> Arbitrary CPs
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
