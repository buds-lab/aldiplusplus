#!/bin/bash

# The argument used is the name of the config file including extension

# pre-process data
python preprocess_modeling.py $1

# training
python train_lgb_meter.py --file $1

# evaluation
python predict_lgb_meter.py --file $1

# blend
# on notebook forecasting_sandbox.ipynb
