from scipy import stats
import math
import torch
#import stumpy
import pyscamp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
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
                 aldi_name='aldi_gmm_dyn_none_both',
                 ):

        """
        Parameter
        ----------
        df_meters: 
            sorted NxM dataframe with M buildings and N rows with hourly
            timestamp as indices
        df_metadata: 
            dataframe with metadata regarding the buildings
        m: 
            hourly window size, one day = 24
        col_id:
            string name of the column with building ids in df_meters and df_metadata
        site_id:
            id of the current portfolio being analyzed
        meter_id:
            id of the current sensor reading being analyzed
        verbose:
            boolean value to enable debugging printing
        gpu:
            TODO: text
        hourly_processing:
            TODO: text
        aldi_name:
            TODO: text
        """

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

        # set auxiliary variables
        self.list_all_bdg = df_meters.columns.values

        # placeholder for upcoming class variables
        self.mp_adj = None
        self.mp_ind = None

        self.df_result = None
        self.num_readings = None
        self.num_buildings = None

        self.df_result_meta = None

        self.df_test = None
        self.df_test_det = None # placeholder

        # start the engine
        self.pipeline()
    
    def pipeline(self):
        
        if self.verbose:
            print(f'Start ALDI. hourly = {self.hourly}')
        ##### EXECUTE ALDI
        #### STEP 1: get mp-values and -indices
        self.mp_adj, self.mp_ind = self.get_mp()
        
        #### STEP 2: select midnight mp-values and base dataframe
        self.df_result, self.num_readings, self.num_buildings = self.data_reconstruction()
        self.df_result_meta = self.add_metadata()
        
        #### STEP 4: run one KS-tests
        self.df_ks_test = self.ks_test()

        #### STEP 5: Classification of the results of the stat test  
        ####         (Initiated by the user from the outside)
        # self.df_test_det = self.get_result_df()

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

        Returns:
        ----------
        df_result: pandas.DataFrame
            text
        num_readings: int
            text
        num_buildings: int
            text
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
            print(f'Updating the combined values with building names ' + 
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
            print(f'Prepared MP values:\n{df_mp}')
            print(f'Shape midnight results:')
            print(f'raw:  {df_e.shape}')
            print(f'mp:   {df_mp.shape}')
            print(f'mpi:  {df_mpind.shape}\n')
            
        return df_e, df_mp, df_mpind

    def add_metadata(self):
        """
        Combines the processed dataframe with matrix profile calculation
        alongside the metadata file

        Returns:
        ----------
        df_result_meta: pandas.DataFrame
            text
        """

        df_result_meta = self.df_result.merge(self.df_metadata,
                                              on=self.col_id)

        if self.verbose:
            print(f'Merging available metadata:\n{df_result_meta.head()}')
        
        return df_result_meta
        
    def ks_test(self):
        """
        Computes an statistical test for each daily distribution

        Returns:
        ----------
        ks_test: pandas.DataFrame
            text
        """

        reference_dist = self.daytype_dist()
        
        if self.hourly:
            curr_freq = 'H'
        else:
            curr_freq = 'D'

        ks_test = pd.DataFrame(columns=['D','p'], 
                               index=pd.date_range(start=self.base_timestamps[0], 
                                                   end=self.base_timestamps[-1],
                                                   freq=curr_freq)
                               )

        if self.verbose:
            print(f'CAUTION: curr_freq: {curr_freq}')
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

    ####################################################################
    #                                                                  #
    #      |||   Methods that are called from the outside    |||       #
    #      VVV                                               VVV       #
    ####################################################################

    def set_gmm_model(self, gmm_data='D', gmm_max_comp=10):
        self.gmm_data = gmm_data
        self.gmm_max_comp = gmm_max_comp

        self.gm_model, self.gmm_components = self._gmm_train()

    def get_result_df(self, forecast_out=False):
        """
        Calculates the discords
        """

        # dynamic share calculation
        max_gauss_mean = self.gmm_components['gauss_mean'].max()
        share_comp_of_interest = 1 - max_gauss_mean

        abs_comp_of_interest = math.trunc(  share_comp_of_interest 
                                         * self.gm_model.n_components)
        sorted_gmm_components = self.gmm_components.sort_values('gauss_mean').copy()
        special_gmm_comp = sorted_gmm_components[:abs_comp_of_interest]
        
        if self.verbose:
            print(f'Share components of interest: {share_comp_of_interest}')
            print(f'Number components of interest: {abs_comp_of_interest}')

        gmm_proba = self.gm_model.predict_proba(
                                self.df_ks_test.D.values.reshape(-1,1))
        df_gmm_proba = pd.DataFrame(gmm_proba, index= self.df_ks_test.index)
        df_gmm_proba['max_comp'] = df_gmm_proba.idxmax(axis='columns')

        if self.gmm_data == 'D':
            gmm_proba = self.gm_model.predict_proba(
                                        self.df_ks_test.D.values.reshape(-1,1))
            df_gmm_proba = pd.DataFrame(gmm_proba, index= self.df_ks_test.index)
            df_gmm_proba['max_comp'] = df_gmm_proba.idxmax(axis='columns')

            # *Important comparison* - The max_comp must be inside the list
            # for the day to be classified as a "non-discord day"
            df_gmm_proba['is_discord'] = np.where(df_gmm_proba['max_comp'].isin(
                                    special_gmm_comp.component), 0, 1)
        else:
            gmm_proba = self.gm_model.predict_proba(
                                        self.df_ks_test.p.values.reshape(-1,1))
            df_gmm_proba = pd.DataFrame(gmm_proba, index= self.df_ks_test.index)
            df_gmm_proba['max_comp'] = df_gmm_proba.idxmax(axis='columns')
            
            # *Important comparison* - The max_comp must be inside the list
            # for the day to be classified as a "non-discord day"
            df_gmm_proba['is_discord'] = np.where(df_gmm_proba['max_comp'].isin(
                                    special_gmm_comp.component), 1, 0)
        
        df_is_discord = pd.DataFrame(index=df_gmm_proba.index)
        df_is_discord['is_discord'] = df_gmm_proba['is_discord']

        # prepare index and column for resulting dataframes
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

    def get_result_using_threshold(self, 
                                   ks_type='D',
                                   threshold=0.6,
                                   forecast_out=False):
        """
        Method offers additional possibility to get a discore 
        classification (predicted). For this purpose, all time points 
        at which a predefined threshold is exceeded are classified as 
        discord and vice versa.

        Parameters:
        ----------
        ks_test: str , required
            Describes which result type of the ks test should be used
        treshold: float , required
            Describes the threshold to be used to distinguish between 
            discord and non-discord
        forecast_out: bool , required
            This parameter controls the formatting of the return type. 
            If False, the resulting dataframe will have columns 
            identifying the different buildings of the site. The index 
            consists of timestamps. 
            If True, the previously described results are formatted 
            into a single column result. The building ID and timestamp 
            are then keys and have their own columns.

        Returns:
        ----------
        df_result: pandas.DataFrame
            see description at parameter 'forecast_out'

        """
        if self.hourly:
            curr_freq = 'H'
        else:
            curr_freq = 'D'

        df_is_discord = pd.DataFrame(
            columns=['is_discord'], 
            index=pd.date_range(start=self.base_timestamps[0], 
                                end=self.base_timestamps[-1],
                                freq=curr_freq)
            )

        df_is_discord['is_discord'] = np.where(
                                    self.df_ks_test[ks_type] < threshold, 1, 0)

        # prepare index and column for resulting dataframes
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
        
        if (forecast_out & (not self.hourly)):
            df_hourly_is_discord = pd.DataFrame(
                index=pd.date_range(start=self.base_timestamps[0], 
                                    end=self.base_timestamps[-1],
                                    freq='H')
            )

            # copy daily dataframe to hourly dataframe
            df_hourly_is_discord['day'] = df_hourly_is_discord.index.date
            df_is_discord.index = df_is_discord.index.date
            df_hourly_is_discord = df_hourly_is_discord.join(
                                        df_is_discord,
                                        on='day', how='left')
            df_hourly_is_discord = df_hourly_is_discord.drop(['day'], axis=1)
            df_result = df_hourly_is_discord.astype('int8')
        else:
            df_result = df_is_discord


        if forecast_out:
            df_result['timestamp'] = df_result.index
            df_result = df_result.melt(id_vars=['timestamp'], 
                                       var_name='building_id', 
                                       value_name='is_discord')

        return df_result

    def plot_true_n_gmm(self, df_true_labels, df_ks_results=None, gmm=None):
        """
        method does something
        """
        if df_ks_results is None:
            df_ks_results = self.df_ks_test
        if gmm is None:
            gmm = self.gm_model

        df_true_labels_day = df_true_labels.groupby(df_true_labels.index.date).max()

        df_ks_results_D = df_ks_results[['D']]
        df_ks_results_D_spez = pd.DataFrame(index=df_ks_results_D.index,
                                            columns=df_true_labels_day.columns)
        
        for col in df_ks_results_D_spez.columns:
            df_ks_results_D_spez[col] = df_ks_results_D['D']

        assert (df_true_labels_day.shape == df_ks_results_D_spez.shape)

        df_D_discord = pd.DataFrame(index=df_ks_results_D.index, 
                                     columns=df_true_labels_day.columns)
        df_D_non_discord = pd.DataFrame(index=df_ks_results_D.index, 
                                        columns=df_true_labels_day.columns)
        for col in df_D_discord.columns:
            df_D_discord[col] = np.where(df_true_labels_day[col] == 1,
                                          df_ks_results_D_spez[col],
                                          math.nan)
            df_D_non_discord[col] = np.where(df_true_labels_day[col] == 0,
                                             df_ks_results_D_spez[col],
                                             math.nan)

        #### HERE THE PLOTTING BEGINNS ###
        x_values = np.linspace(0, 1, 1000)

        logprob = gmm.score_samples(x_values.reshape(-1, 1))
        responsibilities = gmm.predict_proba(x_values.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        number_plot_rows = math.ceil(df_D_discord.shape[1]/2)

        figure, axes = plt.subplots(nrows=number_plot_rows, 
                                    ncols=2, 
                                    figsize=(22, 4*number_plot_rows))
        figure.patch.set_facecolor('white')
        figure.subplots_adjust(top=0.97)
        figure.suptitle(f'D-values of discord and non-discord days from \
                          site {self.site_id}',
                        fontsize=20)

        next_odd = True
        for num, df_col in enumerate(df_D_discord.columns):
            if next_odd:
                plot_col = 0
                next_odd = False
            else:
                plot_col = 1
                next_odd = True
        
            plot_row_num = math.floor(num/2)
    
            try:
                axes[plot_row_num, plot_col].hist([df_D_non_discord[df_col], df_D_discord[df_col]], 
                                                  100,
                                                  density=True,
                                                  histtype='stepfilled', 
                                                  alpha=0.7, 
                                                  label=['non-discord','discord'])
            except AttributeError:
                print('ooouw that hurts')
    
            axes[plot_row_num,plot_col].plot(x_values, pdf, '-k')
            axes[plot_row_num,plot_col].plot(x_values, pdf_individual)
            axes[plot_row_num,plot_col].set_title(f'Information about {df_col}')
            axes[plot_row_num,plot_col].legend(loc='upper right')

        figure.savefig(f'img/D_visualization/{self.aldi_name}/site_{self.site_id}.png',
                       format='PNG')
        plt.clf()

    def plot_true_one_gmm(self, 
                          df_true_labels,
                          agg_type=None,
                          gmm_data='D',
                          df_ks_results=None,
                          gmm=None):
        """
        method creates a plot. Two histograms are shown on the plot. 
        The first histogram shows the distribution of the D-values of 
        the (true) discords. The second histogram shows the distribution 
        of the D-values of the (true) non-discords. 
        Furthermore, the components of the GMM are also visualized.

        Parameters:
        ----------
        df_true_labels: pandas.DataFrame , required
            text
        gmm_data : str , required
            text
        df_ks_results: pandas.DataFrame , optional
            text
        gmm: sklearn.mixture.GaussianMixture , optional
            text

        Returns:
        ----------
            Method saves a plot.
        """
        
        if df_ks_results is None:
            df_ks_results = self.df_ks_test
        if gmm is None:
            gmm = self.gm_model
        if agg_type is None:
            path_prefix = f'img/D_visualization/{self.aldi_name}/'
        else:
            path_prefix = f'img/D_visualization/{self.aldi_name}/{agg_type}/'

        assert (df_true_labels.shape[0] == df_ks_results.shape[0]), 'same length please'

        df_ks_results_D = df_ks_results[[gmm_data]]
        df_ks_results_D_spez = pd.DataFrame(index=df_ks_results_D.index,
                                            columns=df_true_labels.columns)
        for col in df_ks_results_D_spez.columns:
            df_ks_results_D_spez[col] = df_ks_results_D[gmm_data]

        assert (df_true_labels.shape == df_ks_results_D_spez.shape)

        df_D_discord = pd.DataFrame(index=df_ks_results_D.index, 
                                    columns=df_true_labels.columns)
        df_D_non_discord = pd.DataFrame(index=df_ks_results_D.index, 
                                        columns=df_true_labels.columns)

        for col in df_D_discord.columns:
            df_D_discord[col] = np.where(df_true_labels[col] == 1,
                                          df_ks_results_D_spez[col],
                                          math.nan)
            df_D_non_discord[col] = np.where(df_true_labels[col] == 0,
                                             df_ks_results_D_spez[col],
                                             math.nan)

        list_D_non_discord = df_D_non_discord.values.flatten() 
        list_D_discord = df_D_discord.values.flatten()
        cleaned_list_D_non_discord = \
            [x for x in list_D_non_discord if str(x) != 'nan']
        cleaned_list_D_discord = \
            [x for x in list_D_discord if str(x) != 'nan']

        #### HERE THE PLOTTING BEGINNS ###
        fontsize=22


        # first # ONLY HISTOGRAMMS
        x_values = np.linspace(0, 1, 1000)

        logprob = gmm.score_samples(x_values.reshape(-1, 1))
        responsibilities = gmm.predict_proba(x_values.reshape(-1, 1))
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        figure, axes = plt.subplots(nrows=1, 
                                    ncols=1, 
                                    figsize=(18, 6))
        figure.patch.set_facecolor('white')
        figure.suptitle(f'Histogram of the Distance Values from the KS Test'
                        f' (Site {self.site_id})',
                        fontsize=fontsize+4)
    
        try:
            axes.hist([cleaned_list_D_non_discord, 
                       cleaned_list_D_discord], 
                      100,
                      density=False,
                      histtype='stepfilled', 
                      alpha=0.7,
                      range=(0,1),
                      stacked=True,
                      label=['non-discord','discord'])
        except AttributeError:
            print('ooouw that hurts')
    
        #axes.plot(x_values, pdf, '-k')
        #axes.plot(x_values, pdf_individual)
        axes.legend(loc='upper right', prop={'size': fontsize})
        axes.tick_params(labelsize=fontsize)

        plt.xlabel('Distance Value', fontsize=fontsize+2)
        plt.ylabel('Frequency', fontsize=fontsize+2)

        figure.savefig(f'{path_prefix}site_{self.site_id}.png',
                       format='PNG')
        plt.clf()


        # second # COMBINED PLOT
        figure, axes = plt.subplots(nrows=1, 
                                    ncols=1, 
                                    figsize=(18, 6))
        figure.patch.set_facecolor('white')
        figure.suptitle(f'Histogram and Trained GMM of the Distance Values from the KS Test'
                        f' (site {self.site_id})',
                        fontsize=fontsize+4)
    
        try:
            axes.hist([cleaned_list_D_non_discord, 
                       cleaned_list_D_discord], 
                      100,
                      density=True,
                      histtype='stepfilled', 
                      alpha=0.7, 
                      range=(0,1),
                      stacked=True,
                      label=['non-discord','discord'])
        except AttributeError:
            print('ooouw that hurts')
    
        axes.plot(x_values, pdf, '-k')
        axes.plot(x_values, pdf_individual)
        axes.legend(loc='upper right', prop={'size': fontsize})
        axes.tick_params(labelsize=fontsize)

        plt.xlabel('Distance Value', fontsize=fontsize+2)
        plt.ylabel('Density', fontsize=fontsize+2)

        figure.savefig(f'{path_prefix}density_site_{self.site_id}.png',
                       format='PNG')
        plt.clf()



        # third  # ONLY GMM
        figure, axes = plt.subplots(nrows=1, 
                                    ncols=1, 
                                    figsize=(18, 6))
        figure.patch.set_facecolor('white')
        figure.suptitle(f'Trained GMM of the Distance Values from the KS Test'
                        f' (site {self.site_id})',
                        fontsize=20)
    
        axes.plot(x_values, pdf, '-k')
        axes.plot(x_values, pdf_individual)
        axes.legend(loc='upper right', prop={'size': fontsize})
        axes.tick_params(labelsize=fontsize)

        plt.xlabel('Distance Value', fontsize=fontsize+2)
        plt.ylabel('Density', fontsize=fontsize+2)

        figure.savefig(f'{path_prefix}gmm_site_{self.site_id}.png',
                       format='PNG')
        plt.clf()
        plt.close('all')


        # third  # GMM + unlabeled Histo
        figure, axes = plt.subplots(nrows=1, 
                                    ncols=1, 
                                    figsize=(18, 6))
        figure.patch.set_facecolor('white')
        figure.suptitle(f'Trained GMM of the Distance Values from the KS Test'
                        f' (site {self.site_id})',
                        fontsize=fontsize+4)
    
        axes.plot(x_values, pdf, '-k')
        axes.plot(x_values, pdf_individual)
        axes.legend(loc='upper right', prop={'size': fontsize})
        axes.tick_params(labelsize=fontsize)

        plt.xlabel('Distance Value', fontsize=fontsize+2)
        plt.ylabel('Density', fontsize=fontsize+2)

        figure.savefig(f'{path_prefix}gmm_site_{self.site_id}.png',
                       format='PNG')
        plt.clf()


        # forth # COMBINED PLOT (UNLABELED)
        figure, axes = plt.subplots(nrows=1, 
                                    ncols=1, 
                                    figsize=(18, 6))
        figure.patch.set_facecolor('white')
        figure.suptitle(f'Histogram and Trained GMM of the Distance Values from the KS Test'
                        f' (site {self.site_id})',
                        fontsize=fontsize+4)
    
        try:
            axes.hist((cleaned_list_D_non_discord + cleaned_list_D_discord), 
                      100,
                      density=True,
                      histtype='stepfilled', 
                      alpha=0.7, 
                      range=(0,1),
                      stacked=True)
        except AttributeError:
            print('ooouw that hurts')
    
        axes.plot(x_values, pdf, '-k')
        axes.plot(x_values, pdf_individual)
        axes.legend(loc='upper right', prop={'size': fontsize})
        axes.tick_params(labelsize=fontsize)

        plt.xlabel('Distance Value', fontsize=fontsize+2)
        plt.ylabel('Density', fontsize=fontsize+2)

        figure.savefig(f'{path_prefix}unlabeled_site_{self.site_id}.png',
                       format='PNG')
        plt.clf()


        plt.close('all')

    def plot_common_pD( self, 
                        df_true_labels,
                        agg_type='',
                        df_ks_results=None):
        """
        method does something

        Parameters:
        ----------
        df_true_labels: pandas.DataFrame , required
            text
        agg_type: string , optional
            text
        df_ks_results: pandas.DataFrame , optional
            text
        gmm: sklearn.mixture.GaussianMixture , optional
            text

        Returns:
        ----------
            Method saves a plot.
        """
        
        if df_ks_results is None:
            df_ks_results = self.df_ks_test
            # columns = ['D','p']
            # index = timestamps, DatetimeIndex, either hourly or daily
        if agg_type is None:
            path_prefix = f'img/D_visualization/{self.aldi_name}/'
        else:
            path_prefix = f'img/D_visualization/{self.aldi_name}/{agg_type}/'

        assert (df_true_labels.shape[0] == df_ks_results.shape[0]), 'same length please'

        df_all_dat = pd.DataFrame(columns=['date', 'D', 'p', 'label'])
        
        # build one dataframe with following structure:
        #     columsn = ['date', 'D', 'p', 'label']
        #     index = range(N) (N =   number of ks-results 
        #                           * number of buildings within the site)
        # dataframe units all labesl & KS results within single columns
        for label_col in df_true_labels.columns:
            # Prepare true label df
            df_label_tmp = df_true_labels[[label_col]].copy()
            df_label_tmp['date'] = df_label_tmp.index
            df_label_tmp = df_label_tmp.reset_index(drop=True)

            # Prepare KS test result 
            df_ks_tmp = df_ks_results.copy()
            df_ks_tmp['date'] = df_ks_tmp.index
            df_ks_tmp['date'] = df_ks_tmp['date'].dt.date
            df_ks_tmp = df_ks_tmp.reset_index(drop=True)

            df_both_tmp = df_ks_tmp.merge(df_label_tmp, how='inner', on='date')
            df_both_tmp = df_both_tmp.rename(columns={label_col: 'label'})

            df_all_dat = df_all_dat.append(df_both_tmp, ignore_index=True)

        # Create 2D plots
        self._creat_single_pD_2D(df_all_dat, path_prefix)
        self._creat_common_pD_2D(df_all_dat, path_prefix)
        # Create 3D plots
        self._creat_pD_3D(df_all_dat, path_prefix)

        plt.close('all')

    ####################################################################
    #      |||    Support methods for access from outside    |||       #
    #      VVV                                               VVV       #
    ####################################################################

    def _creat_single_pD_2D(self, df_all_dat, path_prefix):
        # Data preparation
        df_ks_true_discord = df_all_dat.query('label == 1')
        df_ks_true_non_discord = df_all_dat.query('label == 0')

        # set plotting parameters
        colors = ['red', 'blue']
        markers = ['o', '^']
        labels = ['true_discord', 'true_non_discord']

        #### FIRST PLOT: DISCORD SCATTER    
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, 
            title=(f'Joint visualisation (2D) of D and p values; '
                   f'only discords (site {self.site_id})'))

        ax.set_xlabel('D-value')
        ax.set_ylabel('p-value')
        ax.set_xlim(0, 1)     #D
        ax.set_ylim(0, 1)     #p

        scatter_dis = ax.scatter(x=df_ks_true_discord['D'],
                                 y=df_ks_true_discord['p'],
                                 color=colors[0], 
                                 alpha=0.3,
                                 marker=markers[0])

        ax.legend(  [scatter_dis], 
                    [labels[0]],
                    numpoints = 1)

        fig.savefig(f'{path_prefix}pD_2D_discord_site_{self.site_id}.png',
                    format='PNG')
        plt.clf()


        #### SECOND PLOT: NON DISCORD SCATTER 
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, 
            title=(f'Joint visualisation (2D) of D and p values; '
                   f'only non-discords (site {self.site_id})'))

        ax.set_xlabel('D-value')
        ax.set_ylabel('p-value')
        ax.set_xlim(0, 1)     #D
        ax.set_ylim(0, 1)     #p

        scatter_non_dis = ax.scatter(x=df_ks_true_non_discord['D'],
                                     y=df_ks_true_non_discord['p'],
                                     color=colors[1], 
                                     alpha=0.3,
                                     marker=markers[1])

        ax.legend(  [scatter_non_dis], 
                    [labels[1]],
                    numpoints = 1)

        fig.savefig(f'{path_prefix}pD_2D_non_discord_site_{self.site_id}.png',
                    format='PNG')
        plt.clf()

    def _creat_common_pD_2D(self, df_all_dat, path_prefix):
        # Data preparation
        df_ks_true_discord = df_all_dat.query('label == 1')
        df_ks_true_non_discord = df_all_dat.query('label == 0')

        # set plotting parameters
        colors = ['red', 'blue']
        markers = ['o', '^']
        labels = ['true_discord', 'true_non_discord']

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, 
            title=(f'Joint visualisation (2D) of D and p values '
                   f'(site {self.site_id})'))

        ax.set_xlabel('D-value')
        ax.set_ylabel('p-value')
        ax.set_xlim(0, 1)     #D
        ax.set_ylim(0, 1)     #p

        scatter_dis = ax.scatter(x=df_ks_true_discord['D'],
                                 y=df_ks_true_discord['p'],
                                 color=colors[0], 
                                 alpha=0.3,
                                 marker=markers[0])
        scatter_non_dis = ax.scatter(x=df_ks_true_non_discord['D'],
                                     y=df_ks_true_non_discord['p'],
                                     color=colors[1], 
                                     alpha=0.3,
                                     marker=markers[1])

        ax.legend(  [scatter_dis, scatter_non_dis], 
                    [labels[0], labels[1]],
                    numpoints = 1)

        fig.savefig(f'{path_prefix}pD_2D_site_{self.site_id}.png',
                    format='PNG')
        plt.clf()

    def _creat_pD_3D(self, df_all_dat, path_prefix):
        # Data preparation
        df_ks_true_discord = df_all_dat.query('label == 1')
        df_ks_true_non_discord = df_all_dat.query('label == 0')

        # set plotting parameters
        scale_x = 1         # true discord labels
        scale_y = 3         # D value
        scale_z = 3         # p value
        colors = ['red', 'blue']
        markers = ['o', '^']
        labels = ['true_discord', 'true_non_discord']

        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, 
                            title=(f'Joint visualisation of D and p '
                                   f'values (site {self.site_id})'), 
                            projection='3d')

        ax.set_xlim(0, 1)
        ax.set_xticks([0,1])    # true discord labels
        ax.set_ylim(0, 1)       # D value
        ax.set_zlim(0, 1)       # p value

        ax.set_xlabel('Discord Label')
        ax.set_ylabel('D-value')
        ax.set_zlabel('p-value')

        # scale the plot if wanted - did not look quite good
        #ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 2]))

        # create discord scatter
        ax.scatter(ys=df_ks_true_discord['D'],
                   zs=df_ks_true_discord['p'],
                   xs=df_ks_true_discord['label'],
                   color=colors[0], 
                   alpha=0.3,
                   marker=markers[0])
        # create non discord scatter
        ax.scatter(ys=df_ks_true_non_discord['D'],
                   zs=df_ks_true_non_discord['p'],
                   xs=df_ks_true_non_discord['label'],
                   color=colors[1], 
                   alpha=0.3,
                   marker=markers[1])

        # Add legend - need some hidden plots -.-
        scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0], marker = markers[0])
        scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker = markers[1])
        ax.legend([scatter1_proxy, scatter2_proxy], [labels[0], labels[1]], numpoints = 1)

        fig.savefig(f'{path_prefix}pD_3D_site_{self.site_id}.png',
                    format='PNG')
        plt.clf()

    def _gmm_train(self):
        """
        trains several GM models based on the given data (train_data) 
        and returns the best one (evaluated by AIC) (best_gmm)
        Also returns a dataframe with a summary of the different 
        GMM components (gmm_components)
        """

        train_data = self._data_for_gmm_training()
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

    def _plot_gmm_results(self, gmm):
        """
        Method prepares a plot. On it you can see the PDF (Probability 
        density function) trained by the given GMM (black line). In 
        addition, the profiles of the individual components of the GMM 
        are displayed (colored lines). 

        If the original data on which the GMM was trained are also 
        given, a histogram is shown in the background.
        """

        x_values = np.linspace(0, 1, 1000)
        y_values = self._data_for_gmm_training()

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

    def _data_for_gmm_training(self):
        if self.gmm_data == 'D':
            y_values = self.df_ks_test.D
        else:
            y_values = self.df_ks_test.p

        return y_values
