import os
import argparse
import glob
import numpy as np 
import pandas as pd 
from functools import partial
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from utils import (
    load_data,
    rmsle,
    timer,
    GeneralizedMeanBlender
)


parser = argparse.ArgumentParser(description="")

parser.add_argument("--file", help="Configuration file")

if __name__ == "__main__":

    args = parser.parse_args()

    # load config file from CLI
    with open(str(args.file), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    algorithm = config["algorithm"]
    output_location = config["output_location"]

    MODEL_LIST = [
        f"output/{algorithm}/lgb-split_meter-no_normalization.npy",
    ]
    
     # load test data
    with timer("load test data"):
        test = load_data("test_clean", algorithm=algorithm, output_location=output_location)

    # load predictions
    with timer("load predictions"):
        preds_matrix = [np.load(x) for x in MODEL_LIST if ".npy" in x]
        preds_matrix = np.vstack(preds_matrix).T
    #     preds_matrix[preds_matrix < 0] = 0

    #  blend predictions
    with timer("blend predictions"):
        gmb = GeneralizedMeanBlender()
        gmb.p = 0.11375872112626925
        gmb.c = 0.99817730007820798
        gmb.weights = [1]
        test_preds = 0.99576627605010293*np.expm1(gmb.transform(np.log1p(preds_matrix)))

    # create submission            
    with timer("create submission"):            
        subm = load_data("sample_submission", data_location=data_location)
        subm.loc[test.meter == 0, "meter_reading"] = test_preds
        subm.loc[subm.meter_reading < 0, "meter_reading"] = 0

    # save data
    with timer("save data"):
        subm.to_csv(f"output/{algorithm}/final_submission.csv", index=False)
