import argparse
import glob
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import (
    Logger,
    timer,
    rmsle,
    load_data,
    make_dir,
)

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--normalize_target",
    action="store_true",
    help="If True then normalize the meter_reading by dividing by log1p(square_feet).",
)

parser.add_argument("--file", help="Configuration file")

FEATURES = [
    # building meta features
    "square_feet",
    "year_built",
    "floor_count",
    # cat cols
    "building_id",
    "site_id",
    "primary_use",
    "hour",
    "weekday",
    "weekday_hour",
    "building_weekday_hour",
    "building_weekday",
    "building_hour",
    # raw weather features
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    # derivative weather features
    "air_temperature_mean_lag7",
    "air_temperature_std_lag7",
    "air_temperature_mean_lag73",
    "air_temperature_std_lag73",
    # time features
    "weekday_x",
    "weekday_y",
    "is_holiday",
    # target encoding features
    # "gte_meter_building_id_hour", "gte_meter_building_id_weekday",
]

CAT_COLS = [
    "building_id",
    "site_id",
    "primary_use",
    "hour",
    "weekday",
    "weekday_hour",
    "building_weekday_hour",
    "building_weekday",
    "building_hour",
]

DROP_COLS = [
    # time columns
    "year",
    "timestamp",
    "hour_x",
    "hour_y",
    # weather extremum
    "air_temperature_min_lag7",
    "air_temperature_max_lag7",
    "air_temperature_min_lag73",
    "air_temperature_max_lag73",
    # first-order gte
    # "gte_hour",
    # "gte_weekday",
    # "gte_month",
    # "gte_building_id",
    # "gte_meter",
    # "gte_meter_hour",
    # "gte_primary_use",
    # "gte_site_id",
    ## second-order gte
    # "gte_meter_weekday",
    # "gte_meter_month",
    # "gte_meter_building_id",
    # "gte_meter_primary_use",
    # "gte_meter_site_id",
    # month columns
    "month_x",
    "month_y",
    "building_month",  # "month",
    # "gte_meter_building_id_month",
]


if __name__ == "__main__":

    args = parser.parse_args()

    # load config file from CLI
    with open(str(args.file), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    algorithm = config["algorithm"]
    output_location = config["output_location"]

    with timer("Loading data"):
        test = load_data(
            "test_clean", algorithm=algorithm, output_location=output_location
        )
        test.drop(DROP_COLS, axis=1, inplace=True)

    with timer("Preprocesing"):
        for x in CAT_COLS:
            test[x] = test[x].astype("category")

        #if args.normalize_target:
        #    target_encode_cols = [x for x in test.columns if "gte" in x]
        #    test[target_encode_cols] = test[target_encode_cols] / np.log1p(
        #        test[["square_feet"]].values
        #    )

    with timer("Predicting"):
        # get base file name
        test_preds = np.zeros(len(test))
        for m in [0]: #range(4): 

            # create sub model path
            if args.normalize_target:
                sub_model_path = (
                    f"models/{algorithm}/lgb-split_meter/target_normalization/meter_{m}"
                )
            else:
                sub_model_path = f"models/{algorithm}/lgb-split_meter/no_normalization/meter_{m}"

            # remove indices not in this meter
            X = test.loc[test.meter == m, FEATURES]
            print(f"split meter {m}: test size {len(X)}")

            # load models
            model_list = glob.glob(f"{sub_model_path}/*")
            assert len(model_list) != 0, "No models to load"

            # predict
            msg = (
                f"Predicting for meter {m} - models# {len(model_list)}, test# {len(X)}"
            )
            with timer(msg):
                preds = 0
                for model_name in model_list:
                    model = lgb.Booster(model_file=model_name)
                    with timer(f" Model {model_name}"):
                        preds += model.predict(X) / len(model_list)
                test_preds[test.meter == m] = preds

        # invert target transformation
        if args.normalize_target:
            test_preds *= np.log1p(test.square_feet)

        test_preds = np.expm1(test_preds)

        # correct site 0
        test_preds[(test.site_id == 0) & (test.meter == 0)] *= 3.4118
        test_preds[test_preds < 0] = 0

    # save data
    make_dir(f"output/{algorithm}")
    if args.normalize_target:
        np.save(f"output/{algorithm}/lgb-split_meter-target_normalization", test_preds)
    else:
        np.save(f"output/{algorithm}/lgb-split_meter-no_normalization", test_preds)
