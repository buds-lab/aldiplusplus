import os
import time
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import datetime

from datetime import datetime
from contextlib import contextmanager, redirect_stdout
from functools import partial
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta
import matplotlib.pyplot as plt


def save_variable(file_name, variable):
    pickle.dump(variable, open(file_name + ".pickle", "wb"))


def load_variable(filename):
    with open(filename + ".pickle", "rb") as f:
        return pickle.load(f)


def zero_is_discord(df_site):
    """Flag all 0 readings as outliers. Only for electricity meter"""

    df_hourly_is_discord = pd.DataFrame(index=df_site.index)
    all_bdg = df_site.columns
    columns = [f"is_discord_{x}" for x in all_bdg]

    # replace NaN with 0
    df_site = df_site.replace(0, np.nan)

    # hand waving specialization (caution) of discords for all bdgs
    for col, bdg in zip(columns, all_bdg):
        df_hourly_is_discord[col] = np.where(df_site[bdg] == 0, 1, 0)

    return df_hourly_is_discord

def get_data_chunks(dataframe, folds=12):
    """
    Function that splits the current dataframe into "k" folds of data where
    1 chunk is seen as train set and k-1 folds as test set.
    Initially it is assume the "k" represents the number of months since the
    time window of "dataframe" is one year.
    This function also performs min-max scaling to the data.
    
    Keyword arguments:
    dataframe -- DataFrame with 24 features as hourly readings and "n" buildings
        with a DateIndex
    folds -- Int specifiying the number of folds required
    
    Returns:
    train_k_folds -- list with "k" elements where each elements is the respective
        train dataframe for the current fold
    test_k_folds -- list with "k" elements where each elements is the respective
        test dataframe for the current fold
    """
    
    df = dataframe.iloc[:, 0:-1].copy()
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_scaled = pd.DataFrame(scaler.transform(df), index=df.index)
    df_scaled["building_id"] = dataframe["building_id"]
    
    train_k_folds = []
    test_k_folds = []

    for k in range(1, folds + 1):
        if folds == 12:
            train_k = df_scaled[df_scaled.index.month == k]
            test_k = df_scaled[df_scaled.index.month != k]
        elif folds == 6 or folds == 4 or folds == 3 or folds == 2 :
            months = list(range(1+(k-1)*int(12/folds), k*int(12/folds) + 1))

            train_k = df_scaled[df_scaled.index.month.isin(months)]
            test_k = df_scaled[~df_scaled.index.month.isin(months)]
        else:
            print("Sorry, that number hasn't been implemented yet")
            return

        # months should not overlap, intersection has to be an empty set
        assert set.intersection(set(train_k.index.month), set(test_k.index.month)) == set()
        
        train_k_folds.append(train_k)
        test_k_folds.append(test_k)

    return train_k_folds, test_k_folds, scaler

def create_daily_plots_for_single_bldg(self, 
                                       bldg_id, 
                                       df_bldg_meter,
                                       df_ks_test_results,
                                       list_days):
    '''
    '''
    matplotlib.use('TkAgg')

    print(f'Started plotting for building {bldg_id}')
    
    # define standard scaler
    std_scaler = StandardScaler()

    df_curr_building = df_bldg_meter.copy()
    #list_all_days = pd.to_datetime(df_ks_test_results['timestamp']).dt.date

    save_path = f'img/daily_plots/bldg_{str(bldg_id).zfill(4)}'
    os.makedirs(save_path, exist_ok=True)
    
    for single_day in list_days:
        curr_row_ks_test_df = df_ks_test_results[  (df_ks_test_results['timestamp'] == single_day.strftime('%Y-%m-%d'))
                                                 & (df_ks_test_results['building_id'] == bldg_id)].index[0]
        
        current_D_val = df_ks_test_results.loc[curr_row_ks_test_df, 'D']
        current_D_val = round(current_D_val, 6)
        current_p_val = df_ks_test_results.loc[curr_row_ks_test_df, 'p']
        current_p_val = round(current_p_val, 6)
        
        ### PREPARE PLOTTING DATA
        df_curr_building_curr_day = df_curr_building[  
                (df_curr_building['timestamp'] >= single_day.strftime('%Y-%m-%d')) 
              & (df_curr_building['timestamp'] < (single_day + timedelta(days=1)).strftime('%Y-%m-%d'))].copy()
        
        if df_curr_building_curr_day.empty:
            continue
        
        df_curr_building_curr_day['norm_meter_reading'] = \
            std_scaler.fit_transform(df_curr_building_curr_day[['meter_reading']])
        
        ### START PLOTTING
        fig,ax=plt.subplots()
        fig.patch.set_facecolor('white')

        df_curr_building_curr_day.plot(y='meter_reading',
                                       x='timestamp',
                                       ax=ax,
                                       kind='line',
                                       figsize=(20,5),
                                       legend=False,
                                       title=('Raw / Normalised load profile '
                                              f'of bldg {bldg_id} '
                                              f'on day {single_day}'),
                                       color="red")
        ax.set_ylabel('Raw load profile', fontsize=12, color="red")

        ax2=ax.twinx()
        df_curr_building_curr_day.plot(y='norm_meter_reading',
                                       x='timestamp',
                                       ax=ax2,
                                       kind='line',
                                       figsize=(18,5),
                                       legend=False,
                                       color='blue')
        ax2.set_ylabel('Normalised load profile', fontsize=12, color='blue')
        complete_path = f'{save_path}/{single_day.strftime("%Y-%m-%d")}_D-{current_D_val}_p-{current_p_val}.png'
        fig.savefig(complete_path,
                    format='png',
                    dpi=50)
        plt.close('all')
        
        df_ks_test_results.loc[curr_row_ks_test_df, 'plot_path'] =  complete_path

def forecasting_barplot(
    dict_algo,
    metric='rmsle',
    plot_name='forecasting',
    x_labels=['ALDI', 'ALDI++'],
    figsize=(16,32),
    ylim=(2,3),
    fontsize=40
):
    """Plot the chosen forecasting RMSE based on different discord detectors"""
     
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        x=list(dict_algo.keys()),
        y=list(dict_algo.values()),
        orient='v', ax=ax
    )
        
    ax.set_ylabel("RMSLE", fontsize=fontsize)
    ax.set_xlabel("Discord detectors", fontsize=fontsize)
    ax.set_xticklabels(x_labels)
    ax.tick_params(length=20, direction="inout", labelsize=fontsize)
    plt.ylim(ylim)

    plt.hlines(
        xmin=0 - 0.5, 
        xmax=len(list(dict_algo.keys()))-0.5, 
        y=list(dict_algo.values())[0], 
        colors='r', 
        linewidth=3 # vertical line at position 0
    )
    plt.tight_layout()

    fig.savefig(f'img/barplot_comparison-{plot_name}.png', format='PNG')

def forecasting_bubble(
    dict_algo,
    plot_name='forecasting',
    y_labels=['ALDI', 'ALDI++'],
    figsize=(16,32),
    xlim=(2,3),
    fontsize=40
):
    """Plot the chosen forecasting RMSE based on different discord detectors"""
    
    # prepare dataframe
    df_discord_detectors = pd.DataFrame.from_dict(dict_algo, orient='index')
    
    fig, _ = plt.subplots(1, 1, figsize=figsize)
    ax = sns.scatterplot(
        data=df_discord_detectors,
        x="rmsle",
        y=df_discord_detectors.index,
        size="time",
        alpha=1,
        sizes={
            1: 20,
            8: 80,
            32: 320,
            40: 400,
            480: 4800,
        },
        size_order=[8, 40, 480],
        clip_on=False
    )
    ax.set_xlabel("RMSLE", fontsize=fontsize)
    ax.set_ylabel("Discords labeled by", fontsize=fontsize)
    ax.set_yticklabels(y_labels)
    ax.tick_params(length=20, direction="inout", labelsize=fontsize)
    ax.legend(
        title="Computation \n time (min)",
        title_fontsize=fontsize-3,
        fontsize=fontsize-3,
        frameon=False,
        bbox_to_anchor=(1, 0.85),
        ncol=1,
    )
    ax.margins(y=0.1)
    plt.grid()
#     ax.set(frame_on=False)
    plt.xlim(xlim)
    plt.tight_layout()
    
    fig.savefig(f'img/bubbleplot_comparison-{plot_name}.png', format='PNG')
    
###############################################################################
# functions from here on are taken from:
# https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/ashrae/utils.py


@contextmanager
def timer(name):
    print(f"{datetime.now()} - [{name}] ...")
    t0 = time.time()
    yield
    print(f"{datetime.now()} - [{name}] done in {time.time() - t0:.0f} s\n")


def reduce_mem_usage(df, skip_cols=[], verbose=False):
    """Reduce memory usage in a pandas dataframe
    Based on this great kernel:
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    """
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in np.setdiff1d(df.columns, skip_cols):
        if df[col].dtype != object:  # Exclude strings

            # print column type
            if verbose:
                print("******************************")
                print("Column: ", col)
                print("dtype before: ", df[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            if verbose:
                print("min for this col: ", mn)
                print("max for this col: ", mx)

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = df[col] - asint
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)

            if verbose:
                print("dtype after: ", df[col].dtype)
                print("******************************")

    # Print final result
    if verbose:
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return df, NAlist


def load_data(
    data_name,
    algorithm="baseline",
    data_location="data",
    discord_location="data/outliers",
    discord_file="bad_meter_readings.csv",
    output_location="data/preprocessed",
):
    """Loads and formats data"""

    # raw
    if data_name == "train":
        return pd.read_csv(f"{data_location}/train.csv")

    if data_name == "test":
        return pd.read_csv(f"{data_location}/test.csv")

    if data_name == "input":
        return load_data("train", data_location=data_location), load_data(
            "test", data_location=data_location
        )

    # clean
    if data_name == "train_clean":
        return pd.read_pickle(f"{output_location}/train_clean_{algorithm}.pkl")

    if data_name == "test_clean":
        return pd.read_pickle(f"{output_location}/test_clean_{algorithm}.pkl")

    if data_name == "clean":
        return (
            load_data(
                "train_clean", output_location=output_location, algorithm=algorithm
            ),
            load_data(
                "test_clean", output_location=output_location, algorithm=algorithm
            ),
        )

    #     # debug 1000
    #     if data_name == "train_clean_debug_1000":
    #         return pd.read_pickle("data/preprocessed/train_clean_debug_1000.pkl")

    #     if data_name == "test_clean_debug_1000":
    #         return pd.read_pickle("data/preprocessed/test_clean_debug_1000.pkl")

    #     if data_name == "clean_debug_1000":
    #         return load_data("train_clean_debug_1000"), load_data("test_clean_debug_1000")

    #     if data_name == "leak_debug_1000":
    #         return pd.read_pickle("data/preprocessed/leak_debug_1000.pkl")

    #     # debug 10000
    #     if data_name == "train_clean_debug_10000":
    #         return pd.read_pickle("data/preprocessed/train_clean_debug_10000.pkl")

    #     if data_name == "test_clean_debug_10000":
    #         return pd.read_pickle("data/preprocessed/test_clean_debug_10000.pkl")

    #     if data_name == "clean_debug_10000":
    #         return load_data("train_clean_debug_10000"), load_data("test_clean_debug_10000")

    #     if data_name == "leak_debug_10000":
    #         return pd.read_pickle("data/preprocessed/leak_debug_10000.pkl")

    # raw weather
    if data_name == "train_weather":
        return pd.read_csv(f"{data_location}/weather_train.csv")

    if data_name == "test_weather":
        return pd.read_csv(f"{data_location}/weather_test.csv")

    if data_name == "weather":
        return load_data("train_weather", data_location=data_location), load_data(
            "test_weather", data_location=data_location
        )

    # discord/outliers (rows to drop)
    if data_name == "discord":
        return pd.read_csv(f"{discord_location}/{discord_file}")

    # meta
    if data_name == "meta":
        return pd.read_csv(f"{data_location}/building_metadata.csv")

    # submissions
    if data_name == "sample_submission":
        return pd.read_csv(f"{data_location}/sample_submission.csv")


class Logger(object):
    """Save a string line(s) to a file."""

    def __init__(self, file_path, mode="w", verbose=False):
        self.file_path = file_path
        self.verbose = verbose
        open(file_path, mode=mode)

    def append(self, line, print_line=None):
        if print_line or self.verbose:
            print(line)
        with open(self.file_path, "a") as f:
            with redirect_stdout(f):
                print(line)


def make_dir(dir_name):
    """Create a directory if it doesn"t already exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_validation_months(n_months):
    validation_months_list = [
        np.arange(i + 1, i + 2 + n_months - 1)
        for shift in range(n_months)
        for i in range(shift, 12 + shift, n_months)
    ]
    validation_months_list = [(x - 1) % 12 + 1 for x in validation_months_list]
    return validation_months_list


def get_daily_resolution(df_hourly_data, agg_method="logic_or"):
    """
    The method takes a dataframe with an hourly resolution and
    transforms it into a dataframe with a daily resolution. All data
    points of a day are aggregated with an aggregation function.

    Parameters
    -----------
    df_hourly_data: pandas.DataFrame, required
        Dataframe with is_discord information (coded in 0: non-discord
        and 1: discord) per building with hourly resolution. The index
        consists of hourly timestamps. The columns identify the
        individual buildings with the following scheme
        'is_discord_{bldg_id}'.
    agg_method: str, optional
        String describing the function to be used to aggregate the
        24 values per day. Default is a logic_OR, i.e. if one hour of a
        day is marked as discord, the whole day is considered as
        Discord.

    Returns
    -----------
    df_daily_data: pandas.DataFrame
        Dataframe with is_discord information (coded in 0: non-discord
        and 1: discord) per building with daily resolution. The index
        consists of daily timestamps. The columns identify the
        individual buildings with the following scheme
        'is_discord_{bldg_id}'.

    """

    # STEP 1 - CALCULATE DAILY SUMS
    df_step_1 = df_hourly_data.groupby(df_hourly_data.index.date).sum()

    # STEP 2 - MARKING DISCORD
    discord_cond = {
        "logic_or": (df_step_1 >= 1),
        "logic_and": (df_step_1 == 24),
        "majority": (df_step_1 >= 12),
        "majority_plus": (df_step_1 >= 15),
    }
    assert agg_method in discord_cond.keys()
    discord_agg_func = discord_cond.get(agg_method)
    df_step_2 = df_step_1.where(~discord_agg_func, other=1)

    # STEP 3 - MARKING NON-DISCORD
    non_discord_cond = {
        "fill_zero": (df_step_2 != 1),
    }
    non_discord_agg_func = non_discord_cond.get("fill_zero")
    df_step_3 = df_step_2.where(~non_discord_agg_func, other=0)

    # STEP 4 - FINALIZATION
    df_daily_data = df_step_3.astype("int8")

    return df_daily_data


def rmsle(x, y):
    x = np.log1p(x)
    y = np.log1p(y)
    return np.sqrt(mean_squared_error(x, y))


class GeneralizedMeanBlender:
    """Combines multiple predictions using generalized mean"""

    def __init__(self, p_range=(0, 1), random_state=42):
        """
        Args:
            p_range: Range of the power in the generalized mean. Defalut is (0,2).
            random_state: Seed for the random number generator.
        Returns: GeneralizedMeanBlender object
        """
        self.p_range = p_range
        self.random_state = random_state
        self.p = None
        self.c = None
        self.weights = None

    def _objective(self, trial, X, y):

        # create hyperparameters
        p = trial.suggest_uniform(f"p", *self.p_range)
        c = trial.suggest_uniform(f"c", 0.95, 1.05)
        weights = [trial.suggest_uniform(f"w{i}", 0, 1) for i in range(X.shape[1])]

        # blend predictions
        blend_preds, total_weight = 0, 0
        if p == 0:
            for j, w in enumerate(weights):
                blend_preds += w * np.log1p(X[:, j])
                total_weight += w
            blend_preds = c * np.expm1(blend_preds / total_weight)
        else:
            for j, w in enumerate(weights):
                blend_preds += w * X[:, j] ** p
                total_weight += w
            blend_preds = c * (blend_preds / total_weight) ** (1 / p)

        # calculate mean squared error
        return np.sqrt(mean_squared_error(y, blend_preds))

    def fit(self, X, y, n_trials=10):
        # optimize objective
        obj = partial(self._objective, X=X, y=y)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(sampler=sampler)
        study.optimize(obj, n_trials=n_trials)
        # extract best weights
        if self.p is None:
            self.p = [v for k, v in study.best_params.items() if "p" in k][0]
        self.c = [v for k, v in study.best_params.items() if "c" in k][0]
        self.weights = np.array([v for k, v in study.best_params.items() if "w" in k])
        self.weights /= self.weights.sum()

    def transform(self, X):
        assert (
            self.weights is not None and self.p is not None
        ), "Must call fit method before transform"
        if self.p == 0:
            return self.c * np.expm1(np.dot(np.log1p(X), self.weights))
        else:
            return self.c * np.dot(X ** self.p, self.weights) ** (1 / self.p)

    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)
