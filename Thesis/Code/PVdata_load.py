# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:05:57 2022

@author: joris
"""

import pvlib_helpers as pvgis
import pandas as pd
from darts import TimeSeries

def solardata(Lat, Long, Tilt, Ori, Peak, Loss, Resample):
    # Import data from pvgis (PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM)
    data1 = pvgis.get_pvgis_hourly(Lat,Long,surface_tilt = Tilt, surface_azimuth = Ori, pvcalculation=True, peakpower = Peak, loss=Loss, raddatabase='PVGIS-SARAH2')
    # Transform to dataframes
    df1 = pd.DataFrame(data1[0])
    # Take only load
    load = df1[['P']]
    # Localize
    load = load.tz_localize(None)
    # Resample if needed
    loadRe = load.resample(Resample).mean()
    # Convert to timeseries
    series = TimeSeries.from_dataframe(loadRe)
    
    return series

def solardf(Lat, Long, Tilt, Ori, Peak, Resample, start = None):
    # Import data from pvgis (PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM)
    data = pvgis.get_pvgis_hourly(Lat,Long,start = start,surface_tilt = Tilt, surface_azimuth = Ori, pvcalculation=True, peakpower = Peak)
    # Transform to dataframes
    solardf = pd.DataFrame(data[0]).tz_localize(None)
    #lag1_cols = ['poa_dir_lag','solar_ele_lag','temp_air_lag']
    #lag24_cols = ['poa_dir_lag24','solar_ele_lag24','temp_air_lag24']
    #drop_cols = ['poa_direct','solar_elevation', 'temp_air','wind_speed','poa_sky_diffuse','poa_ground_diffuse','Int']

    #for i in range(len(lag1_cols)):
    #    solardf[lag1_cols[i]] = solardf[drop_cols[i]].shift(1)
    #    solardf[lag24_cols[i]] = solardf[drop_cols[i]].shift(24)

    #solardf['lag1_P'] = solardf['P'].shift(1)
    #solardf['lag24_P'] = solardf['P'].shift(24)
    #solardf.drop(drop_cols, axis = 1, inplace = True)
    #solardf = solardf.dropna()
    
    return solardf

def sitesdata(SiteName):
    # Read the csv files
    data0 = pd.read_csv("..\\..\\Data\\forecast50-1.csv", header = None, parse_dates = [1], index_col = [1])
    data1 = pd.read_csv("..\\..\\Data\\forecast50-2.csv", header = None, parse_dates = [1], index_col = [1])
    data2 = pd.read_csv("..\\..\\Data\\forecast50-3.csv", header = None, parse_dates = [1], index_col = [1])
    # Concatenate the files
    data = pd.concat([data0, data1, data2], axis=0)
    data.rename(columns = {0:'Site', 2:'load'}, inplace = True)
    data.index.names = ['Date']
    # Pick a site for analysis
    sitedata = data.where(data['Site'] == SiteName)
    
    return sitedata

latitude = 52.31
longitude =  4.89
surface_tilt = 35
surface_azimuth = -67
peakpower = 2.48 # in KW
periodicity = 'H'
start = None

solardf = solardf(latitude, longitude, surface_tilt, surface_azimuth, peakpower, periodicity, start)
