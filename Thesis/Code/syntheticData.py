# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:28:15 2022

@author: joris
"""

import PVdata_load as load

import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt

from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler

def SyntheticDataPrepped(pvgisDataset, validation = True, trainTestSplit = 0.8, featurisationOption = 1, exogenous = None):
    
    """
    This function takes the dataset from pvgis and transforms it into a usable train and test set.
    There are 3 featurisation options:
        1. No covariates, only the  PV data
        2. Hour and month as cyclic patterns using sin and cosin functions, exogenous variables as given
        3. Hour and month as One Hot Encoded variables, exogenous variables as given
    """
    
    # For option 4 we would include an if statement and not remove    
    
    solar = pvgisDataset['P'].astype(np.float32)
    
    if featurisationOption == 1:
        print("Splitting the data, only weather variables, no cyclic variables\n")
        
        print("preparing train-test split of target variable\n")
        # create timeseries (Darts format)
        pv_series = TimeSeries.from_series(solar)

        # Create training and validation sets:
        train, test = pv_series.split_after(trainTestSplit)
        transformer = Scaler()
        train_transformed = transformer.fit_transform(train)
        series_transformed = transformer.transform(pv_series)

        if(validation == True):
            val, test = test.split_after(pd.to_datetime('2018-01-01 00:11:00'))
            val_transformed = transformer.transform(val)
            test_transformed = transformer.transform(test)
        else:
            test_transformed = transformer.transform(test)

        weather_covariates = pvgisDataset[exogenous]
        covariates = TimeSeries.from_dataframe(weather_covariates).astype(np.float32) 
        
        cov_train, cov_test = covariates.split_after(trainTestSplit)
        cov_transform = Scaler()
        cov_train = cov_transform.fit_transform(cov_train)
        covariates = cov_transform.transform(covariates)

        if(validation == True):
            cov_val, cov_test = cov_test.split_after(pd.to_datetime('2018-01-01 00:11:00'))
            cov_val = cov_transform.transform(cov_val)
            cov_test = cov_transform.transform(cov_test)
        else:
            cov_val = []
            cov_test = cov_transform.transform(cov_test)
    
    elif featurisationOption == 2:
        print("Using the following exogenous variables: " + str(exogenous) + "\n")
        
        # create timeseries (Darts format)
        pv_series = TimeSeries.from_series(solar)

        # Create training and validation sets:
        train, test = pv_series.split_after(trainTestSplit)
        transformer = Scaler()
        train_transformed = transformer.fit_transform(train)
        series_transformed = transformer.transform(pv_series)

        if(validation == True):
            val, test = test.split_after(pd.to_datetime('2018-01-01 00:11:00'))
            val_transformed = transformer.transform(val)
            test_transformed = transformer.transform(test)
        else:
            val_transformed = []
            test_transformed = transformer.transform(test)

        # create month and year covariate series
        hour_series = datetime_attribute_timeseries(
            pd.date_range(start=pv_series.start_time(), freq=pv_series.freq_str, periods=len(pvgisDataset)),
            attribute="hour",
            cyclic=True
            ).astype(np.float32)

        #month_series = datetime_attribute_timeseries(
        #    hour_series, attribute="month", cyclic=True).astype(np.float32)


        weather_covariates = pvgisDataset[exogenous]
        weather_series = TimeSeries.from_dataframe(weather_covariates).astype(np.float32) 
        covariates = weather_series
        covariates = weather_series.stack(hour_series)
        #covariates = covariates.stack(month_series)
        
        cov_train, cov_test = covariates.split_after(trainTestSplit)
        cov_transform = Scaler()
        cov_train = cov_transform.fit_transform(cov_train)
        covariates = cov_transform.transform(covariates)

        if(validation == True):
            cov_val, cov_test = cov_test.split_after(pd.to_datetime('2018-01-01 00:11:00'))
            cov_val = cov_transform.transform(cov_val)
            cov_test = cov_transform.transform(cov_test)
        else:
            cov_val = []
            cov_test = cov_transform.transform(cov_test)



    else:
        print("Using the following exogenous variables: " + str(exogenous) + "\n")
        
        print("preparing train-test split of target variable and one-hot encoded time variables\n")
        # create timeseries (Darts format)
        pv_series = TimeSeries.from_series(solar)

        # Create training and validation sets:
        train, test = pv_series.split_after(trainTestSplit)
        transformer = Scaler()
        train_transformed = transformer.fit_transform(train)
        series_transformed = transformer.transform(pv_series)

        if(validation == True):
            val, test = test.split_after(pd.to_datetime('2018-01-01 00:11:00'))
            val_transformed = transformer.transform(val)
            test_transformed = transformer.transform(test)
        else:
            val_transformed = []
            test_transformed = transformer.transform(test)
        
        # create month and year covariate series
        hour_series = datetime_attribute_timeseries(
            pd.date_range(start=pv_series.start_time(), freq=pv_series.freq_str, periods=len(pvgisDataset)),
            attribute="hour",
            one_hot=True
            ).astype(np.float32)

        #month_series = datetime_attribute_timeseries(
        #    hour_series, attribute="month", one_hot=True).astype(np.float32)


        weather_covariates = pvgisDataset[exogenous]
        weather_series = TimeSeries.from_dataframe(weather_covariates).astype(np.float32) 
        covariates = weather_series
        covariates = weather_series.stack(hour_series)
        #covariates = covariates.stack(month_series)
        
        cov_train, cov_test = covariates.split_after(trainTestSplit)
        cov_transform = Scaler()
        cov_train = cov_transform.fit_transform(cov_train)
        covariates = cov_transform.transform(covariates)

        if(validation == True):
            cov_val, cov_test = cov_test.split_after(pd.to_datetime('2018-01-01 00:11:00'))
            cov_val = cov_transform.transform(cov_val)
            cov_test = cov_transform.transform(cov_test)
        else:
            cov_val = []
            cov_test = cov_transform.transform(cov_test)

            
    print("dataset ready\n")
    return pv_series, train_transformed, val_transformed, test_transformed, series_transformed, covariates, cov_train, cov_val, cov_test