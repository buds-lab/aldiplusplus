from scipy import stats
import stumpy
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

class ALDI():
    def __init__(self, df_meters, df_metadata, m=24, col_id='building_id', site_id='', meter_id='', verbose=False):
        """
        Args:
            df_meters: sorted NxM dataframe with M buildings and N rows with hourly
                timestamp as indices
            df_metadata: dataframe with metadata regarding the buildings
            m: hourly window size, one day = 24
            col_id: string name of the column with building ids in df_meters and df_metadata
            site_id: id of the current portfolio being analyzed
            meter_id: id of the current sensor reading being analyzed
            verbose: boolean value to enable debugging printing
        """
        self.df_meters =  df_meters.copy()
        self.df_metadata = df_metadata.copy()
        self.base_timestamps = df_meters.copy().index
        self.m = m
        self.col_id = col_id
        self.site_id = site_id
        self.meter_id = meter_id
        self.verbose = verbose
            
        # auxiliary variables needed
        self.name_list = df_meters.columns
        
        ##### execute ALDI
        self.mp_adj, self.mp_ind = self.get_mp() # get matrix profile and indices
        
        # merge information to one single dataframe
        self.df_result, self.num_days, self.num_buildings = self.data_reconstruction()
        self.df_result_meta = self.add_metadata()
        
        # calculate k-test
        self.df_ks_test = self.k_test()
        self.df_ks_test_det = None # placeholder
    
    def zero_coun(self): # TODO: implement
        pass
    
    def get_mp(self):
        """
        Calculates matrix profile and matrix profile indices for a time-stamp 
        sorted dataframe where the columns are buildings from the same site
        and rows are meter readings.
            
        Returns:
            mp_adj: dataframe with the matrix profile values
            mp_ind: dataframe with the matrix profile indices
        """
        mp_adj = pd.DataFrame(columns=self.name_list)
        mp_ind = pd.DataFrame(columns=self.name_list)
        
        for col in self.name_list:
            bldg = self.df_meters[col]
            mp = stumpy.stump(bldg, m=self.m)
            
            # append np.nan to matrix profile to allow plotting against raw data
            madj = np.append(mp[:,0], np.zeros(self.m-1) + np.nan)
            mind = np.append(mp[:,1], np.zeros(self.m-1) + np.nan)
            
            # save mp information
            mp_adj[col] = madj
            mp_ind[col] = mind
    
        return mp_adj, mp_ind
    
    def midnight_mp(self):
        """
        Picks daily matrix profile at midnight
        """
        # use only available timestamps
        df_e = self.df_meters.copy()
        
        df_mp = self.mp_adj.set_index(df_e.index)
        df_mpind = self.mp_ind.set_index(df_e.index)

        df_e_0 = df_e[df_e.index.hour==0]
        df_mp_0 = df_mp[df_mp.index.hour==0]
        df_mpind_0 = df_mpind[df_mpind.index.hour==0]
        
        if self.verbose:
            print(f'Midnight MP values:\n{df_e_0}')
            
        return df_e_0, df_mp_0, df_mpind_0
    
    def data_reconstruction(self):
        """
        Puts together calculated values into one single dataframe
        """
        df_result = pd.DataFrame(columns=['raw','mp','mp_ind'])

        df_e_0, df_mp_0, df_mpind_0 = self.midnight_mp()
        
        num_days = df_e_0.shape[0]
        num_buildings = df_e_0.shape[1]

        
        print(f'num of days: {num_days}') # debug
        
        
        # combining the matrix profile and indices values
        df_result['raw'] = df_e_0.values.reshape(num_days * num_buildings)
        df_result['mp'] = df_mp_0.values.reshape(num_days * num_buildings)
        df_result['mp_ind'] = df_mpind_0.values.reshape(num_days * num_buildings)

        if self.verbose:
            print(f'Combining raw and calculated values:\n{df_result}')
        
        df_names=[]
        df_dates=[]
        days=[]

        self.year = df_e_0.index[0].year
        self.month = df_e_0.index[0].month
        self.day = df_e_0.index[0].day

        # combining the building names and dates
        for i in range(num_days):
            df_names = np.append(df_names, np.array(self.name_list))
            days = np.append(days, np.ones(len(self.name_list))*i)
        for i in range(len(days)):
            df_dates = df_dates + \
                [dt.datetime(year=self.year,month=self.month,day=self.day) + \
                 dt.timedelta(days=days[i])]

        df_result[self.col_id] = df_names
        df_result['date'] = df_dates
        
        if self.verbose:
            print(f'Updating the combined values with building names and full dates:\n{df_result}')
        
        # combining the breakdown of the dates
        df_month=[]
        df_daytype=[]
        df_day=[]

        for i in range(len(df_result)):
            df_month = np.append(df_month, df_result.date[i].strftime('%b'))
            df_daytype = np.append(df_daytype, df_result.date[i].strftime('%a'))
            df_day = np.append(df_day, df_result.date[i].strftime('%d'))

        df_result['month'] = df_month
        df_result['daytype'] = df_daytype
        df_result['day'] = df_day
         
        if self.verbose:
            print(f'Updating the combined values with broken down dates:\n{df_result}')
        
        return df_result, num_days, num_buildings

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
        """Computes daytype distributions"""
        daytype_dist = {}
        daytype_dist['mon'] = self.df_result.mp[self.df_result.daytype == 'Mon']
        daytype_dist['tue'] = self.df_result.mp[self.df_result.daytype == 'Tue']
        daytype_dist['wed'] = self.df_result.mp[self.df_result.daytype == 'Wed']
        daytype_dist['thu'] = self.df_result.mp[self.df_result.daytype == 'Thu']
        daytype_dist['fri'] = self.df_result.mp[self.df_result.daytype == 'Fri']
        daytype_dist['sat'] = self.df_result.mp[self.df_result.daytype == 'Sat']
        daytype_dist['sun'] = self.df_result.mp[self.df_result.daytype == 'Sun']
        
        return daytype_dist
        
    def k_test(self):
        """Computes k-s test for each daily distribution"""
        daytype_dist = self.daytype_dist() # compute daily distributions
        
        ks_test = pd.DataFrame(columns=['D','p'],
                               index=pd.date_range(pd.datetime(year=self.year, month=self.month, day=self.day), 
                                                   periods=self.num_days))

        for i in pd.date_range(pd.datetime(year=self.year, month=self.month, day=self.day), periods=self.num_days):
            events = self.df_result.mp[self.df_result.date == i]
            
            if i.weekday() == 0:
                test = stats.ks_2samp(events, daytype_dist['mon'])
                ks_test.D[i] = test.statistic
                ks_test.p[i] = test.pvalue

            if i.weekday() == 1:
                test = stats.ks_2samp(events, daytype_dist['tue'])
                ks_test.D[i] = test.statistic
                ks_test.p[i] = test.pvalue

            if i.weekday() == 2:
                test = stats.ks_2samp(events, daytype_dist['wed'])
                ks_test.D[i] = test.statistic
                ks_test.p[i] = test.pvalue

            if i.weekday() == 3:
                test = stats.ks_2samp(events, daytype_dist['thu'])
                ks_test.D[i] = test.statistic
                ks_test.p[i] = test.pvalue

            if i.weekday() == 4:
                test = stats.ks_2samp(events, daytype_dist['fri'])
                ks_test.D[i] = test.statistic
                ks_test.p[i] = test.pvalue

            if i.weekday() == 5:
                test = stats.ks_2samp(events, daytype_dist['sat'])
                ks_test.D[i] = test.statistic
                ks_test.p[i] = test.pvalue

            if i.weekday() == 6:
                test = stats.ks_2samp(events, daytype_dist['sun'])
                ks_test.D[i] = test.statistic
                ks_test.p[i] = test.pvalue

        if self.verbose:
            print(f'K-S test dataframe:\n{ks_test}')

        return ks_test
        
    def get_rejected_days(self):
        """
        Calculates the rejected days at commonly used p-values
        Returns:
            p_nr: dataframe with the total number of rejected days at 
                the given p-value(s)
        """
        
        ks_test = self.df_ks_test.copy()
        p_nr = pd.DataFrame(columns=['p','nr'])
        
        # by default compute commonly used p-values
        p_nr.p = [0.01, 0.05, 0.1, 0.15, 0.2]    
        p_nr.nr = np.zeros(len(p_nr.p))
        
        for i in range(len(p_nr)):
            ks_test['det_aux'] = np.where(ks_test['p'] < p_nr.p[i], 1, 0)
            temp = ks_test
            temp = pd.Series(ks_test.det_aux)
            p_nr.nr[i] = np.sum(temp)

        return p_nr

    def get_discords(self, pvalue=0.01):
        """Calculates the discords at a given p-value"""

        # filter based on pvalue
        ks_test = self.df_ks_test.copy()
        ks_test['det'] = np.where(ks_test['p'] < pvalue, 1, 0)
        discord = ks_test[ks_test['det'] == 1]
        
        # plot
        sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.8)
        plt.figure(figsize=[3, 5])
        sns.boxplot(data=discord['D'], orient='vertical')
        plt.ylim(0,1)
        plt.xlabel(f'Site {self.col_id}')
        plt.ylabel('D')
        plt.savefig(f'img/discords_{pvalue}-{self.site_id}-{self.meter_id}.png', bbox_inches='tight', format='PNG')
        plt.close()

        # sort the dataframe and calculate quantiles
        discord_sort = discord.sort_values(by='D')
        discord_q = self.get_discords_quantiles(discord_sort)
        
        self.df_ks_test_det = ks_test
        
        return discord_sort, discord_q


    def get_result_df(self, p_value=0.01):
        """Calculates the discords at a given p-value"""

        # prepare index and column for resulting dataframes
        hourly_timestamps = self.base_timestamps.copy()
        all_bdg = self.name_list.copy()
        columns = [f'is_discord_{x}' for x in all_bdg]

        # filter based on p_value
        df_daily_is_discord = self.df_ks_test.copy()
        df_daily_is_discord['is_discord'] = np.where(
            df_daily_is_discord['p'] < p_value, 1, 0)
        
        # hand waving specialization (caution) of discords for all bdgs
        for col in columns:
            df_daily_is_discord[col] = df_daily_is_discord['is_discord']

        df_daily_is_discord = df_daily_is_discord.drop(['p', 'D', 'is_discord'], axis=1)

        df_hourly_is_discord = pd.DataFrame(index = hourly_timestamps)

        # copy daily dataframe to hourly dataframe
        df_hourly_is_discord['day'] = df_hourly_is_discord.index.date
        df_daily_is_discord.index = df_daily_is_discord.index.date
        df_hourly_is_discord = df_hourly_is_discord.join(df_daily_is_discord,
                                                         on='day', how='left')
        df_hourly_is_discord = df_hourly_is_discord.drop(['day'], axis=1)
        df_hourly_is_discord = df_hourly_is_discord.astype('int8')

        return df_hourly_is_discord

    
    def get_discords_quantiles(self, discord_sorted):
        """Calculates the IQR discords"""

        df_e = self.df_meters.copy()
        df_e_z = pd.DataFrame(stats.zscore(df_e, axis=0, nan_policy='omit'),index=df_e.index)
            
        for i in discord_sorted.index[-3:]: # why 3?
            discord_temp = df_e_z[i:i + dt.timedelta(hours=self.m-1)] # 23 for daily
#             print(i, self.df_ks_test.D[i], self.df_ks_test.p[i])
            
            discord_q = pd.DataFrame(columns=['q1','q2','q3'],index=discord_temp.index)
            
            for j in range(len(discord_temp)):
                # replaced np.percentile with nanpercentile
                discord_q['q1'][j] = np.nanpercentile(discord_temp.iloc[j,:], 25)
                discord_q['q2'][j] = np.nanpercentile(discord_temp.iloc[j,:], 50)
                discord_q['q3'][j] = np.nanpercentile(discord_temp.iloc[j,:], 75)

            sns.set(style='white', font_scale=1.5)
            plt.figure(figsize=(5,2))
            plt.plot(discord_q.q1, '--', color='tomato')
            plt.plot(discord_q.q2, color='red')
            plt.plot(discord_q.q3, '--', color='tomato')

            plt.yticks([-10,0,10,20,30])
            plt.xticks([])
            plt.ylim(-18,35)
            
            plt.savefig(f'img/discord_quantiles-{self.site_id}-{self.meter_id}.png', bbox_inches='tight', format="PNG")
            plt.close()
        
        return discord_q
        
    def plot_mp_dist(self, variable):
        """
        Plots the matrix profile values according to the selected variable
        """
        
        sns.set(context='notebook',
                style='white',
                palette='deep',
                font='sans-serif',
                font_scale=1.5,
                color_codes=True,
                rc=None)

        if variable == 'day-month':
            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            available_months = list(self.df_result.month.unique())
            
            fig, axes = plt.subplots(len(months), 1, figsize=[20,5*len(months)])
            
            for i, idx in zip(months, range(0, len(months))):
                if i not in available_months:
                    print(f'Month {i} not available on this site')
                    continue
                events = self.df_result[self.df_result.month == i]
                sns.boxplot(x='day', y='mp', data=events, color='lightgray', ax=axes[idx])
                axes[idx].set_title(i)
#                 plt.ylim(-0.5,5.5)
                axes[idx].set_xlim(-1,31)
                axes[idx].set_xlabel('Days of month')
                axes[idx].set_ylabel('Matrix profile')
            fig.tight_layout()

        elif variable == 'daily':
            plt.figure(figsize=[5,5])
            sns.boxplot(data=self.df_result_meta.mp, color='lightgray', orient='vertical')
            plt.xlabel(variable)
            plt.ylabel('Matrix profile')
        else:
            plt.figure(figsize=[10,5])
            if variable == 'daytype':
                sns.boxplot(x=variable, y='mp', data=self.df_result_meta, color='lightgray',
                           order=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            else:
                sns.boxplot(x=variable, y='mp', data=self.df_result_meta, color='lightgray')
            plt.xlabel(variable)
            plt.ylabel('Matrix profile')
#         plt.ylim(-0.5,10)
        
        plt.savefig(f'img/mp_dist_{variable}-{self.site_id}-{self.meter_id}.png', bbox_inches='tight', format='PNG')
        plt.close()
     
    def all_plots_mp(self):
        """Plots all mp distribution variants"""
        # mp distribiution
        self.plot_mp_dist('month')
        self.plot_mp_dist('daytype')
        self.plot_mp_dist('day-month')
        self.plot_mp_dist('primary_use')
        self.plot_mp_dist('daily')

    def plot_ks_test_result(self, value='d'):
        """Visualize k-s test"""
        events = self.df_ks_test.copy()
        
        if value == 'd':
            events = pd.Series(self.df_ks_test.D)
            cmap = "YlGnBu_r"
        elif value == 'p':
            events = pd.Series(self.df_ks_test.p)
            cmap = "Greys_r"
        else:
            events = pd.Series(self.df_ks_test_det.det)
            cmap = "Greys"
        
        fig, ax = calplot.calplot(events,
                                  cmap=cmap,
                                  figsize=[20, 4],
                                  daylabels='MTWTFSS',
                                  linewidth=1,
                                  linecolor='grey',
                                  fillcolor='grey')

        plt.savefig(f'img/ks_test_{value}-{self.site_id}-{self.meter_id}.png', bbox_inches='tight', format='PNG')
        plt.close()
    
    def all_plots_ks(self):
        """Plots all ks-test visualisations"""
        self.plot_ks_test_result('d')
        self.plot_ks_test_result('p')
        self.plot_ks_test_result('det')
    
    def get_motifs(self, n):
        """Plots top n motifs"""
        
        ks_test = self.df_ks_test.copy()
        median_pvalue = ks_test['p'].median()
        motifs = ks_test[ks_test['p'] <= median_pvalue]
        motifs_sorted = motifs.sort_values(by='D', ascending=False)

        # plot distribution
        sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.8)
        plt.figure(figsize=[3, 5])
        sns.boxplot(data=motifs_sorted['D'], orient='vertical')
        plt.ylim(0,1)
        plt.xlabel(f'Site {self.col_id}')
        plt.ylabel('D')
        plt.savefig(f'img/motifs_median-{self.site_id}-{self.meter_id}.png', bbox_inches='tight', format='PNG')
        plt.close()

        # plot motifs
        df_e = self.df_meters.copy()
        df_e_z = pd.DataFrame(stats.zscore(df_e, axis=0),index=df_e.index)
        
        for i in motifs_sorted.index[:n]:
            motif_temp = df_e_z[i:i+dt.timedelta(hours=self.m-1)]
            print(i,ks_test.D[i],ks_test.p[i])
            motif_q = pd.DataFrame(columns=['q1','q2','q3'], index=motif_temp.index)
            for j in range(len(motif_temp)):
                # replaced np.percentile with nanpercentile
                motif_q['q1'][j] = np.nanpercentile(motif_temp.iloc[j,:], 25)
                motif_q['q2'][j] = np.nanpercentile(motif_temp.iloc[j,:], 50)
                motif_q['q3'][j] = np.nanpercentile(motif_temp.iloc[j,:], 75)

            sns.set(style='white', font_scale=1.5)
            plt.figure(figsize=(5,2))
            plt.plot(motif_q.q1, '--', color='grey')
            plt.plot(motif_q.q2, color='k')
            plt.plot(motif_q.q3, '--', color='grey')

            plt.xticks([])
            plt.xlim(i,i + dt.timedelta(hours=23))
            #plt.savefig("Graph" + str(i) +".png", bbox_inches='tight', format="PNG")
            plt.show()
        
        # plot raw data at motif dates
        for i in motifs_sorted.index[:n]:
            sns.set(style='white', font_scale=1.5)
#             print(i,ks_test.D[i],ks_test.p[i])
            plt.figure(figsize=(5,2))
            plt.plot(df_e_z[i:i+dt.timedelta(hours=self.m-1)])
            #plt.yticks([])
            plt.xticks([])
            #plt.xlim(i,i + dt.timedelta(hours=23))
            #plt.savefig("Graph" + str(i) +".png", bbox_inches='tight', format="PNG")
            plt.show()