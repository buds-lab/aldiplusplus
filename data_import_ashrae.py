import numpy as np
import pandas as pd


class DataImportAshrae():
    """
    class provides different methods to import BDG2 data 
    for experiments with Discord Detectors
    """

    def __init__(self):
        """
        method initializes df_all_data
        """
        self.df_all_data = None

    def get_meter_data( self, energy_type_list: list, site_list: list,
                        verbose=False):
        """
        method returns a sorted NxM dataframe with M buildings and N
        rows with hourly timestamp as indices

        Keyword arguments:
        energy_type_list -- List with the requested meter types
        site_list -- List with the requested site ids
        verbose -- enable debugging printing (default False)

        Returns:
        df_result -- dataframe (NxM) (N = #timestamps, M = #buildings)
        with readings at corresponding times at corresponding buildings 
        """
        df_site_meter = self._prepare_and_filter_raw_data(  energy_type_list,
                                                            site_list)

        filter_col = ['meter_reading']
        df_result = self._build_desired_df( df_site_meter, 
                                            filter_col, 
                                            addBuildingID= True,
                                            addDiscordID= False)
        return df_result

    def get_daily_profiles( self, meter_type=0, site_list=[]):
        """
        method returns a sorted Nx24 dataframe with M buildings and 24
        features (hourly readings) and hourly timestamp as indices

        Keyword arguments:
        meter_type --Integer with the requested meter types
        site_list -- List with the requested site ids
    
        Returns:
        df_result -- dataframe (NxM) (N = #buildings x days, 24 = hourly
        readings)
        """

        df_site_meter = DataImportAshrae().get_meter_data([meter_type], site_list)
        building_list = df_site_meter.columns
        
        og_idx = df_site_meter.reset_index()["timestamp"].copy()    
        df_site_meter = df_site_meter.reset_index(drop=True)
        
        day_list = []
        bdg_list = []
        idx_list = []
            
        for bid in building_list:
            for i in range(0, df_site_meter.shape[0], 24):
                day = df_site_meter.loc[i:i+23, bid]
                day_list.append(day)
                idx_list.append(og_idx[i])
                bdg_list.append(bid)

        df_all = pd.DataFrame(np.stack(day_list, axis=0), index=idx_list)
        df_all['building_id'] = bdg_list
        
        # remove days with nan values
        df_all = df_all.dropna(axis=0, how='any')
        assert df_all.isnull().values.any() == False 

        return df_all
    

    def get_labeled_meter_data( self, energy_type_list: list, site_list: list,
                                verbose=False):
        """
        method returns a sorted Nx(2*M) dataframe with M buildings, 
        M corresponding labels (is_discord?) and N rows with 
        hourly timestamp as indices

        Keyword arguments:
        energy_type_list -- List with the requested meter types
        site_list -- List with the requested site ids
        verbose -- enable debugging printing (default False)

        Returns:
        df_result -- dataframe (Nx(2*M)) (N = #timestamps, M = #buildings)
        with readings at corresponding times at corresponding buildings
        and a label that describes if the reading is a discord
        """
        df_site_meter = self._prepare_and_filter_raw_data(  energy_type_list,
                                                        site_list)

        filter_col = ['meter_reading', 'is_discord']
        df_result = self._build_desired_df( df_site_meter, 
                                            filter_col, 
                                            addBuildingID= True, 
                                            addDiscordID= True)

        return df_result


    def get_label_data(self, energy_type_list, site_list, verbose=False):
        """
        method returns a sorted NxM dataframe with M buildings and 
        N rows with hourly timestamp as indices

        Keyword arguments:
        energy_type_list -- List with the requested meter types
        site_list -- List with the requested site ids
        verbose -- enable debugging printing (default False)

        Returns:
        df_result -- dataframe (NxM) (N = #timestamps, M = #buildings)
        with labels (is_discord?) at corresponding times 
        at corresponding buildings 
        """        
        df_site_meter = self._prepare_and_filter_raw_data(  energy_type_list,
                                                        site_list)

        filter_col = ['is_discord']
        df_result = self._build_desired_df( df_site_meter,
                                            filter_col, 
                                            addBuildingID= False,
                                            addDiscordID= True)

        return df_result

    def get_meta_data(self, verbose= False):
        """
        method returns a dataframe with metadata from the BDG2 dataset

        Keyword arguments:
        verbose -- enable debugging printing (default False)

        Returns:
        df_result -- metadata from BDG2
        """
        if self.df_all_data is None:
            self.df_all_data = self._get_raw_data().copy()

        df_result = self.df_all_data.copy()
        df_result = df_result.filter([  'site_id',
                                        'building_id',
                                        'primary_use',
                                        'square_feet',
                                        'year_built',
                                        'floor_count',],
                                     axis=1)
        df_result = df_result.drop_duplicates(subset=[  'site_id',
                                                        'building_id',])

        return df_result

    def get_timestamps(self, verbose= False):
        """
        Method returns a dataframe with all reading times

        Keyword arguments:
        verbose -- enable debugging printing (default False)

        Returns:
        df_result -- dataframe with all reading times
        """
        if self.df_all_data is None:
            self.df_all_data = self._get_raw_data().copy()
        
        df_result = self.df_all_data.copy()
        df_result = df_result.filter(['timestamp'], axis=1)
        df_result = df_result.drop_duplicates()

        return df_result

    def get_timestamps_buildings(self, resolution='H'):
        """
        TODO
        """
        assert (resolution in ['H', 'D']), ('Make sure that the '
                                            'resolution is either "H" '
                                            '(hourly) or "D" (daily)')
        
        if self.df_all_data is None:
            self.df_all_data = self._get_raw_data().copy()

        df_result = self.df_all_data.copy()
        df_result = df_result.filter(['timestamp', 
                                      'building_id',
                                      'meter',
                                     ], axis=1)

        if resolution == 'D':
            df_result['date'] = df_result['timestamp'].dt.date
            df_result = df_result.drop(['timestamp'], axis=1)
            df_result = df_result.drop_duplicates()
            df_result = df_result.rename(columns={'date': 'timestamp'})
            df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])

        assert (True not in df_result.duplicated().unique()), (
                'Something went wrong. At this point, duplicates must'
                'no longer appear in the dataframe df_result!')

        return df_result

    def get_vacation_data(self, site_id:int, verbose=False):
        """
        TODO
        """
        excel_vacations = pd.ExcelFile(r'data/holidays/Great Energy Predictor III Schedule Data Collection.xlsx')
        dict_site_sheet = {
            1:{'id':1,
               'name':'site1',
               'sheet':'University College London',
              },
            2:{'id':2,
               'name':'site2',
               'sheet':'Arizona State',
              },
            4:{'id':4,
               'name':'site4',
               'sheet':'University of California Berkel',
              },
            14:{'id':14,
                'name':'site14',
                'sheet':'Princeton University',
               },
            15:{'id':15,
                'name':'site15',
                'sheet':'Cornell',
               },
        }
        list_available_sites = list(dict_site_sheet.keys())

        # check if the request is legal
        assert site_id in list_available_sites, "Only vacation data for sites 1, 2, 4, 14 and 15 can be exported"

        # select necessary data
        df_vacations = pd.read_excel(excel_vacations, dict_site_sheet[site_id]['sheet'])
        list_columns = list(df_vacations.columns)
        list_columns.remove('Date')
        # calculate is_normal and is_discord column
        df_vacations['Label 1'] = df_vacations['Label 1'].astype('category')
        df_vacations['is_normal'] = pd.get_dummies(df_vacations, 
                                                   columns=['Label 1']
                                                  )['Label 1_Regular']
        assert (df_vacations['is_normal'].sum() 
               == df_vacations[df_vacations['Label 1'] == 'Regular'].shape[0])
        df_vacations['is_discord'] = df_vacations['is_normal'].replace({0:1, 1:0})
        # clean up dataframe
        df_vacations = df_vacations.drop(list_columns, axis=1)
        df_vacations = df_vacations.drop('is_normal', axis=1)
        # filtering the necessary data
        df_vacations = df_vacations.set_index('Date')
        df_vacations = df_vacations.loc['2016-01-01':'2016-12-31']

        return df_vacations

    def _get_raw_data(self, verbose= False):
        assert self.df_all_data is None, "The data has already been loaded"

        # prepare base data
        #(ashrae energy predictor + winning solution s data)
        str_path_prefix = 'data/ashrae-energy-prediction/'
        df_meters = pd.read_csv(str_path_prefix + 'train.csv', 
                                parse_dates= True)
        df_weather = pd.read_csv(   str_path_prefix + 'weather_train.csv', 
                                    parse_dates= True)
        df_metadata = pd.read_csv(  str_path_prefix + 'building_metadata.csv',
                                    parse_dates= True)
        df_discord_labels = pd.read_csv('data/outliers/bad_meter_readings.csv',
                                        dtype= 
                                            {'is_bad_meter_reading': np.int64})
        
        df_discord_labels = df_discord_labels.rename(
                                columns={'is_bad_meter_reading': 'is_discord'})

        # merging the files
        df_all_data = df_meters.merge(  df_metadata, 
                                        on= 'building_id', how= 'left')
        df_all_data = df_all_data.merge(df_weather, 
                                        on= ['site_id', 'timestamp'],
                                        how= 'left')
        df_all_data = df_all_data.merge(df_discord_labels, left_index=True,
                                        right_index=True, how='outer')
        df_all_data.timestamp = pd.to_datetime(df_all_data.timestamp)

        return df_all_data


    def _filter_sites(self, df_base, site_list, verbose=False):
        df_reduced = df_base[(df_base['site_id'].isin(site_list))] 
        return df_reduced


    def _filter_energy_type(self, df_base, energy_type_list, verbose=False):
        df_reduced = df_base[(df_base['meter'].isin(energy_type_list))] 
        return df_reduced


    def _correction_power(self, df_base, verbose=False):
        assert not df_base[(df_base['meter'] == 0) 
                            & (df_base['site_id'] == 0)].empty,(
            "Nothing needs to be corrected in this section of the data")

        # from the rank-1 solution:
        # https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/master/solutions/rank-1/scripts/02_preprocess_data.py
        df_base[(df_base['meter']== 0)
                    & (df_base['site_id']== 0)].meter_reading.mul(0.2931)
        return df_base


    def _prepare_and_filter_raw_data(   self, energy_type_list: list, 
                                        site_list: list):
        if self.df_all_data is None:
            self.df_all_data = self._get_raw_data()

        df_site_meter = self._filter_sites(self.df_all_data.copy(), site_list)
        df_site_meter = self._filter_energy_type(df_site_meter, energy_type_list)

        if (0 in energy_type_list) and (0 in site_list):
            df_site_meter = self._correction_power(df_site_meter)

        return df_site_meter


    def _build_desired_df(  self, df_site_meter, filter_col: list, 
                            addBuildingID=False, addDiscordID=False):
        df_timestamps = pd.DataFrame(
                            {'timestamp': self.df_all_data.timestamp.unique()})
        collector_list = [df_timestamps.set_index('timestamp')]

        # restructure
        for building in df_site_meter.building_id.unique():
            # filter all data for one building
            curr_building = df_site_meter[(df_site_meter['building_id'] 
                                            == building)]
            # ensure that all dates are taken into account
            curr_building = curr_building.merge(df_timestamps, how='outer',
                                                on='timestamp')

            # selection of the wanted data and assignment of suitable names
            curr_building = curr_building.filter(   ['timestamp'] + filter_col,
                                                    axis=1)
            if addBuildingID:
                curr_building = curr_building.rename(
                                    columns={'meter_reading': building})
            if addDiscordID:
                curr_building['is_discord'] = curr_building['is_discord'].fillna(1)
                curr_building['is_discord'] = curr_building['is_discord'].astype('int64')
                curr_building = curr_building.rename(
                                    columns={'is_discord':
                                        ('is_discord_' + str(building))})
            
            curr_building = curr_building.set_index('timestamp')
    
            # appending all buildings
            collector_list.append(curr_building)
    
        df_result = pd.concat(collector_list, axis=1)
        df_result = df_result.sort_index()
        return df_result
