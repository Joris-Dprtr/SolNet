import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler

class RawData():
    
    def __init__(
        self,
        dataList):
        
        self.dataList = dataList
        
    def dataTransforming(self):
        
        trainList = []
        covTrainList = []
        testList = []
        covTestList = []
        
        TRAINTESTSPLIT = 0.85
        
        for i in range(len(self.dataList)):
            
            source_data = self.dataList[i][0].astype(np.float32)
        
            #Create a TimeSeries object of the target variable
            target_series = TimeSeries.from_series(source_data['P'])
        
            # Create training and validation sets of the target variable
            train, test = target_series.split_after(TRAINTESTSPLIT)
            transformer = Scaler()
            train = transformer.fit_transform(train)
            test = transformer.transform(test)     

            #Create a TimeSeries object of the target variable
            covariate_series = TimeSeries.from_series(source_data[['poa_direct','poa_sky_diffuse','poa_ground_diffuse','solar_elevation', 'temp_air']])

            hour_series = datetime_attribute_timeseries(
                pd.date_range(start=covariate_series.start_time(), freq=covariate_series.freq_str, periods=len(source_data)),
                attribute="hour",
                cyclic=True
                ).astype(np.float32)

            covariate_series = covariate_series.stack(hour_series)


            # Create training and validation sets of the target variable
            cov_train, cov_test = covariate_series.split_after(TRAINTESTSPLIT)
            transformer_2 = Scaler()
            cov_train = transformer_2.fit_transform(cov_train)
            cov_test = transformer_2.transform(cov_test)

            trainList.append(train)
            covTrainList.append(cov_train)
            testList.append(test)
            covTestList.append(cov_test)

        return trainList, covTrainList, testList, covTestList