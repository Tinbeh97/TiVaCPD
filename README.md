### Time-Varying Correlation Networks for Interpretable Change Point Detection (TiVaCPD)

The models presented in this represetory are mainly applied for biometric application. 
For more information on this work please read [Time-Varying Correlation Networks for Interpretable Change Point Detection]().

## Prerequisites

All codes are written for Python 3.9.16 (https://www.python.org/) on Linux platform. The required packages are in the requiements (TiVaCPD/requirements.txt) file.

### Clone this repository

```
git clone git@github.com:
```

## Previous work

This represetory contains the codes for previous work.

## Datasets 
6 dataset - 3 simulated and 3 real world data 
simulate.py is for simulating data -> Jumping Mean, changing_variance, changing correlation
simulate2.py -> Arbitrary CPs

## TiVaCPD training 

```
python3 experiments.py --data_path ./data/beedance --exp beedance --model_type MMDATVGL_CPD --data_type beedance --penalty_type L2 --ensemble_win True
```
## TiVaCPD hyperparameters tuning 

```
python3 hyp_tune.py --data_path ./data/beedance --exp beedance --model_type MMDATVGL_CPD --data_type beedance --penalty_type L2
```
## Citation

If you find this repository useful, please consider citing the following papers:
