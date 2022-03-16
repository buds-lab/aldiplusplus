import os
import argparse
import glob
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error, mean_squared_error

from utils import (
    save_variable,
    rmsle,
    load_data,
    timer,
    GeneralizedMeanBlender
)

if __name__ == "__main__":
     # load config file from CLI
    with open(str(sys.argv[1]), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    algorithm = config["algorithm"]
    data_location = config["data_location"]
    output_location = config["output_location"]
    
    # ground truth file
    df_bdg = pd.read_csv("data/meters/kaggle/kaggle.csv")
    df_metadata = pd.read_csv("data/metadata/metadata.csv")
    df_metadata = df_metadata[["building_id_kaggle", "site_id_kaggle", "primaryspaceusage"]]
    df_metadata = df_metadata[df_metadata["building_id_kaggle"].notna()]
    df_metadata["building_id_kaggle"] = df_metadata["building_id_kaggle"].astype(int)

    df_bdg = pd.merge(df_bdg, df_metadata, left_on="building_id", right_on="building_id_kaggle")
    df_bdg = df_bdg.sort_values(by="row_id", ascending=True)

    MODEL_LIST = [
        f"output/{algorithm}/lgb-split_meter-no_normalization.npy",
    ]

    # load predictions
    with timer("load predictions"):
        preds_matrix = [np.load(x) for x in MODEL_LIST if ".npy" in x]
        preds_matrix = np.vstack(preds_matrix).T

    # blend predictions
    with timer("blend predictions"):
        gmb = GeneralizedMeanBlender()
        gmb.p = 0.11375872112626925
        gmb.c = 0.99817730007820798
        gmb.weights = [1]
        test_preds = 0.99576627605010293*np.expm1(gmb.transform(np.log1p(preds_matrix)))

    # filter preditions
    with timer("calculate RMSLE"):
        # compare only test year (2017)
        test_preds = test_preds[0:len(df_bdg)]
        df_bdg["test_preds"] = test_preds
        # filter only electricity predictions
        df_bdg = df_bdg[df_bdg["meter"] == 0]
        # replace NaN
        df_bdg["meter_reading"] = df_bdg["meter_reading"].fillna(0)

        # breakdown of results
        dict_results = {}
        # overall
        rmsle_all = rmsle(df_bdg["meter_reading"], df_bdg["test_preds"])
        dict_results['all'] = rmsle_all
        # site-specific
        for site_id in df_bdg["site_id_kaggle"].unique():
            df_bdg_site = df_bdg.copy()
            df_bdg_site = df_bdg_site[df_bdg_site["site_id_kaggle"] == site_id]
            rmsle_site = rmsle(df_bdg_site["meter_reading"], df_bdg_site["test_preds"])
            dict_results[site_id] = rmsle_site
        # PSU-specific
        for psu in df_bdg["primaryspaceusage"].unique():
            df_bdg_psu = df_bdg.copy()
            df_bdg_psu = df_bdg_psu[df_bdg_psu["primaryspaceusage"] == psu]
            rmsle_psu = rmsle(df_bdg_psu["meter_reading"], df_bdg_psu["test_preds"])
            dict_results[psu] = rmsle_psu

        save_variable(f"results/dict_results_forecasting_{algorithm}", dict_results)
        print(dict_results)
