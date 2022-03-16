from scipy import stats
import math
import torch
#import stumpy
import pyscamp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calmap # not working with latest pandas
import calplot
import joypy
import sys
import time
import datetime as dt
from sklearn.mixture import GaussianMixture

class ALDI():
    def __init__(self, 
                 df_meters, 
                 df_metadata, 
                 m=24, 
                 col_id='building_id', 
                 site_id='', 
                 meter_id='',
                 verbose=False, 
                 gpu=False,
                 hourly_processing=False,
                 aldi_name='aldi_gmm_rel_none_daily'):

        self.df_meters =  df_meters.copy()
        self.df_metadata = df_metadata.copy()
        self.base_timestamps = df_meters.copy().index
        self.m = m
        self.col_id = col_id
        self.site_id = site_id
        self.meter_id = meter_id
        self.verbose = verbose
        self.aldi_name = aldi_name
        self.hourly = hourly_processing
        self.cuda = True and gpu if torch.cuda.is_available() else False
        if self.cuda:
            print('Using GPU')

        # set different dates
        self.year = self.df_meters.index[0].year
        self.month = self.df_meters.index[0].month
        self.day = self.df_meters.index[0].day
        self.num_days = int(len(df_meters.index.values)/24)
        self.daily_timestamps = pd.date_range(dt.datetime(year= self.year,
                                                          month= self.month,
                                                          day= self.day),
                                              periods= self.num_days)
        self.hourly_timestamps = pd.date_range(dt.datetime(year= self.year,
                                                          month= self.month,
                                                          day= self.day),
                                              periods= self.num_days * 24,
                                              freq= 'h')

        # set auxiliary variables
        self.list_all_bdg = df_meters.columns.values
        self.num_bdg = len(self.list_all_bdg)


        #  #  #  #  #  #    EXECUTE ALDI    #  #  #  #  #  #
        #### STEP 1: get mp-values and -indices
        self.mp_adj, self.mp_ind = self.get_mp()

        #### STEP 3: build base dataframe 
        ####         (merge information to one single dataframe)
        self.df_result, self.num_readings, self.num_buildings = self.data_reconstruction()
        self.df_result_meta = self.add_metadata()
        
        #### STEP 4: run one KS-tests
        self.df_ks_test = self.ks_test()

        #### STEP 5: train GMM on the D-values
        
        #self.gm_model, self.gmm_components = self.gmm_train()
        #self.plot_gmm_results(self.gm_model)
        
        ### STEP 6: detect discords
        ###         first approach: All values that are most likely to 
        ###         come from the Gaussian bell with the lowest mean 
        ###         value are not discords

        #### STEP 7: post processing
        # currently there is no postprocessing
        # a combination of the different p-values is possible
    
    def set_gmm_model(self, gmm_data='D', gmm_max_comp=10):
        self.gmm_data = gmm_data
        self.gmm_max_comp = gmm_max_comp

        self.gm_model, self.gmm_components = self.gmm_train()
        self.plot_gmm_results(self.gm_model)

    def get_mp(self):
        """
        Calculates matrix profile and matrix profile indices for a time-stamp 
        sorted dataframe where the columns are buildings from the same site
        and rows are meter readings.
            
        Returns:
            mp_adj: dataframe with the matrix profile values
            mp_ind: dataframe with the matrix profile indices
        """
        mp_adj = pd.DataFrame(columns=self.list_all_bdg)
        mp_ind = pd.DataFrame(columns=self.list_all_bdg)
        
        for col in self.list_all_bdg:
            bldg = self.df_meters[col]

            mp_profile, mp_index = pyscamp.selfjoin(bldg, self.m)

            #if self.cuda:
            #    mp = stumpy.gpu_stump(bldg, m=self.m)
            #else:
            #    mp = stumpy.stump(bldg, m=self.m)
            
            # append np.nan to matrix profile to allow plotting against raw data
            #madj = np.append(mp[:,0], np.zeros(self.m-1) + np.nan)
            #mind = np.append(mp[:,1], np.zeros(self.m-1) + np.nan)
            
            madj = np.append(mp_profile, np.zeros(self.m-1) + np.nan)
            mind = np.append(mp_index, np.zeros(self.m-1) + np.nan)

            # save mp information
            mp_adj[col] = madj
            mp_ind[col] = mind
    
        return mp_adj, mp_ind

    def data_reconstruction(self):
        """
        Puts together calculated values into one single dataframe
        """
        df_result = pd.DataFrame(columns=['raw','mp','mp_ind'])

        # Previous get_midnight_values()
        df_e, df_mp, df_mpind = self.prepare_mp_values()

        num_readings = df_e.shape[0]
        num_buildings = df_e.shape[1]

        if self.verbose:
            print(f'num of readings: {num_readings}') # debug

        # combining the matrix profile and indices values
        df_result['raw'] = df_e.values.reshape(num_readings * num_buildings)
        df_result['mp'] = df_mp.values.reshape(num_readings * num_buildings)
        df_result['mp_ind'] = df_mpind.values.reshape(num_readings * num_buildings)

        if self.verbose:
            print(f'Combining raw and calculated values:\n{df_result}')
        
        # combining the building names and dates
        if self.hourly:
            # HOURLY SOLUTION
            df_names = np.tile(self.list_all_bdg, num_readings)
            steps = np.repeat(list(range(num_readings)), len(self.list_all_bdg))
            df_interim_dates = (pd.date_range(start=self.base_timestamps[0], 
                                              end=self.base_timestamps[-1],
                                              freq='H')
                                ).to_pydatetime().tolist()
            df_dates = np.repeat(df_interim_dates, len(self.list_all_bdg))
        else:
            # DAYS SOLUTION
            df_names = np.tile(self.list_all_bdg, num_readings)
            steps = np.repeat(list(range(num_readings)), len(self.list_all_bdg))

            df_interim_dates = (pd.date_range(start=self.base_timestamps[0], 
                                              end=self.base_timestamps[-1],
                                              freq='d')
                                ).to_pydatetime().tolist()
            df_dates = np.repeat(df_interim_dates, len(self.list_all_bdg))

        df_result[self.col_id] = df_names
        df_result['date'] = df_dates
        
        if self.verbose:
            print(f'Updating the combined values with building names' + 
                  f'and full dates:\n{df_result}')
        
        # combining the breakdown of the dates
        df_result['month'] = df_result['date'].dt.strftime('%b')
        df_result['daytype'] = df_result['date'].dt.strftime('%a')
        df_result['day'] = df_result['date'].dt.strftime('%d')
        df_result['hour'] = (df_result['date'].dt.strftime('%H')).astype('int8')
         
        if self.verbose:
            print(f'Updating the combined values with broken down dates:\n{df_result}')
        
        return df_result, num_readings, num_buildings

    def prepare_mp_values(self):
        """
        Picks daily matrix profile at midnight
    
        Returns:
        ----------
        df_e: pandas.DataFrame
            text
        df_mp: pandas.DataFrame
            text
        df_mpind: pandas.DataFrame
            text
        """

        df_e = self.df_meters.copy()
        df_mp = self.mp_adj.set_index(df_e.index)
        df_mpind = self.mp_ind.set_index(df_e.index)

        if not self.hourly:
            df_e = df_e[df_e.index.hour==0]
            df_mp = df_mp[df_mp.index.hour==0]
            df_mpind = df_mpind[df_mpind.index.hour==0]
        
        if self.verbose:
            print(f'Prepared MP values:\n{df_mp}\n')
            print(f'df_e format: {df_e.shape}')
            print(f'df_mp format: {df_mp.shape}')
            print(f'df_mpind format: {df_mpind.shape}')
            
        return df_e, df_mp, df_mpind

    def add_metadata(self):
        """
        Combines the processed dataframe with matrix profile calculation
        alongside the metadata file
        """
        df_result_meta = self.df_result.merge(self.df_metadata, on=self.col_id)

        if self.verbose:
            print(f'Merging available metadata:\n{df_result_meta.head()}')
        
        return df_result_meta

    def daytype_dist(self):
        """
        Computes daytype distributions

        Returns:
        ----------
        daytype_dist: dictionary
            text
        """
        daytype_dist = {}
        weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

        if self.hourly:
            times = list(range(24))
            for curr_day in weekdays:
                daytype_dist[curr_day] = {}
                for curr_time in times:
                    daytype_dist[curr_day][curr_time] = self.df_result.mp[
                                        (self.df_result.daytype == curr_day)
                                      & (self.df_result.hour == curr_time)  ]
        else:
            for curr_day in weekdays:
                daytype_dist[curr_day] = self.df_result.mp[(   
                                        self.df_result.daytype == curr_day)]
        
        return daytype_dist
    
    def ks_test(self):
        """
        Computes KS test for each daily distribution
        """

        reference_dist = self.daytype_dist()

        # prepare resulting datastructure
        curr_columns = ['D','p']

        if self.hourly:
            curr_freq = 'H'
        else:
            curr_freq = 'D'

        ks_test = pd.DataFrame(columns=curr_columns,
                                 index=pd.date_range(
                                                start=self.base_timestamps[0], 
                                                end=self.base_timestamps[-1],
                                                freq=curr_freq)
                                )
        
        if self.verbose:
            print(f'NOTE: curr_freq: {curr_freq}')
            print(f'Starting to fill the ks_test df: \n{ks_test}')

        for i in ks_test.index:
            events = self.df_result.mp[self.df_result.date == i]
            if self.hourly:
                reference = reference_dist[i.strftime('%a')][int(i.strftime('%H'))]
            else:
                reference = reference_dist[i.strftime('%a')]

            test = stats.ks_2samp(events, reference)
            ks_test.D[i] = test.statistic
            ks_test.p[i] = test.pvalue

        if self.verbose:
            print(f'KS test dataframe:\n{ks_test}')

        return ks_test

    def gmm_train(self):
        """
        trains several GM models based on the given data (train_data) 
        and returns the best one (evaluated by AIC) (best_gmm)
        Also returns a dataframe with a summary of the different 
        GMM components (gmm_components)
        """

        train_data = self.data_for_gmm_training()
        y_values = np.array([[val] for val in train_data])

        N = np.arange(1, (self.gmm_max_comp + 1))
        models = [None for i in range(len(N))]
        for i in range(len(N)):
            models[i] = GaussianMixture(N[i]).fit(y_values)
        
        AIC = [m.aic(y_values) for m in models]
        #BIC = [m.bic(y_values) for m in models]
        best_gmm = models[np.argmin(AIC)]

        gmm_components = pd.DataFrame(columns=['component', 
                                               'gauss_mean',
                                               'gauss_covariance'])
        gmm_components['component'] = list(range(0, best_gmm.n_components))
        gmm_components['gauss_mean'] = best_gmm.means_
        gmm_components['gauss_covariance'] = best_gmm.covariances_.reshape(-1,1)

        if self.verbose:
            print(f'calculated GMM') 
            print(f'components:\n {gmm_components}')

        return best_gmm, gmm_components

    def plot_gmm_results(self, gmm):
        """
        Method prepares a plot. On it you can see the PDF (Probability 
        density function) trained by the given GMM (black line). In 
        addition, the profiles of the individual components of the GMM 
        are displayed (colored lines). 

        If the original data on which the GMM was trained are also 
        given, a histogram is shown in the background.
        """

        x_values = np.linspace(0, 1, 1000)
        y_values = self.data_for_gmm_training()

        logprob = gmm.score_samples(x_values.reshape(-1, 1))
        responsibilities = gmm.predict_proba(x_values.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        figure, axes = plt.subplots(1, 1, figsize=(20, 10))
        figure.patch.set_facecolor('white')

        axes.set_title(f'Trained GMM on {self.gmm_data}-values from site {self.site_id}',
                       fontsize=18)
        
        axes.hist(y_values, 100, density=True,
                  histtype='stepfilled', alpha=0.4)
        axes.plot(x_values, pdf, '-k')
        axes.plot(x_values, pdf_individual)

        figure.savefig(f'img/pD_evaluation/{self.gmm_data}-value_aialdi_gmm_s{self.site_id}_m{self.meter_id}_data-{self.gmm_data}.png',
                       format='PNG')
        plt.clf()

    def data_for_gmm_training(self):
        if self.gmm_data == 'D':
            y_values = self.df_ks_test.D
        else:
            y_values = self.df_ks_test.p

        return y_values

    def get_result_df(self, rel_share_left_comp=0.5, forecast_out=False):
        """
        Calculates the discords
        """

        abs_share_left_comp = math.trunc(  rel_share_left_comp 
                                         * self.gm_model.n_components)
        sorted_gmm_components = self.gmm_components.sort_values(
                                                    'gauss_mean').copy()
        special_gmm_comp = sorted_gmm_components[:abs_share_left_comp]

        gmm_proba = self.gm_model.predict_proba(
                                self.df_ks_test.D.values.reshape(-1,1))
        df_gmm_proba = pd.DataFrame(gmm_proba, index=self.df_ks_test.index)
        df_gmm_proba['max_comp'] = df_gmm_proba.idxmax(axis='columns')

        if self.gmm_data == 'D':
            gmm_proba = self.gm_model.predict_proba(
                                        self.df_ks_test.D.values.reshape(-1,1))
            df_gmm_proba = pd.DataFrame(gmm_proba, index=self.df_ks_test.index)
            df_gmm_proba['max_comp'] = df_gmm_proba.idxmax(axis='columns')

            # *Important comparison* - The max_comp must be inside the list
            # for the day to be classified as a "non-discord day"
            df_gmm_proba['is_discord'] = np.where(df_gmm_proba['max_comp'].isin(
                                    special_gmm_comp.component), 0, 1)
        else:
            gmm_proba = self.gm_model.predict_proba(
                                        self.df_ks_test.p.values.reshape(-1,1))
            df_gmm_proba = pd.DataFrame(gmm_proba, index=self.df_ks_test.index)
            df_gmm_proba['max_comp'] = df_gmm_proba.idxmax(axis='columns')
            
            # *Important comparison* - The max_comp must be inside the list
            # for the day to be classified as a "non-discord day"
            df_gmm_proba['is_discord'] = np.where(df_gmm_proba['max_comp'].isin(
                                    special_gmm_comp.component), 1, 0)
        
        df_is_discord = pd.DataFrame(index=df_gmm_proba.index)
        df_is_discord['is_discord'] = df_gmm_proba['is_discord']

        all_bdg = self.list_all_bdg.copy()
        if forecast_out:
            columns = all_bdg
        else:
            columns = [f'is_discord_{x}' for x in all_bdg]

        # hand waving specialization (caution) of discords for all bdgs
        for col in columns:
            df_is_discord[col] = df_is_discord['is_discord']

        df_is_discord = df_is_discord.drop(['is_discord'],
                                       axis=1)

        if forecast_out:
            if not self.hourly:
                hourly_timestamps = self.base_timestamps
                df_hourly_is_discord = pd.DataFrame(index=hourly_timestamps)

                # copy daily dataframe to hourly dataframe
                df_hourly_is_discord['day'] = df_hourly_is_discord.index.date
                df_is_discord.index = df_is_discord.index.date
                df_hourly_is_discord = df_hourly_is_discord.join(df_is_discord,
                                                                 on='day',
                                                                 how='left')
                df_hourly_is_discord = df_hourly_is_discord.drop(['day'], axis=1)
                df_is_discord_hourly = df_hourly_is_discord.astype('int8')
            else:
                df_is_discord_hourly = df_is_discord

            df_is_discord_hourly['timestamp'] = df_is_discord_hourly.index
            df_is_discord_hourly = df_is_discord_hourly.melt(
                                                id_vars=['timestamp'], 
                                                var_name='building_id', 
                                                value_name='is_discord')

            # Exportable variable
            df_is_discord = df_is_discord_hourly

        return df_is_discord