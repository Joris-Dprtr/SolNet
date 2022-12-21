import math
import pandas as pd
import numpy as np

import pvlib_helpers

from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel

from pytorch_lightning.callbacks import EarlyStopping

class SolNet:

    def __init__(
        self,
        latitude,
        longitude,
        peakPower,
        locations = 5,
        start_date = 2005):
        
        
        self.dataGathering(latitude, longitude, peakPower, locations, start_date)
        

    def dataGathering(latitude, longitude, peakPower, locations, start_date = 2005):
        
        km_radius = 50          #The radius around the actual location to find additional locations
        gaus_radius = 0.5       #The covariance for gaussian noise in km on the radius
        precision = 40          #The closeness to trailing the coastline when locations are close to water
        
        data = []
        
        # Make a dataframe for the locations
        lin_loc = np.linspace(0,locations,locations)
        lin_loc_df = pd.DataFrame({"Locations":lin_loc})

        # Make a sin and cosin pattern based on the number of locations
        lin_loc_df["loc_norm"] = 2 * math.pi * lin_loc_df["Locations"] / lin_loc_df["Locations"].max()

        lin_loc_df["sin_loc"] = np.sin(lin_loc_df["loc_norm"])
        lin_loc_df["cos_loc"] = np.cos(lin_loc_df["loc_norm"])

        # Base location

        print('Gathering data from base location...')
        
        try:
            data.append(pvlib_helpers.get_pvgis_hourly(latitude=latitude,
                                          longitude=longitude, 
                                          start = start_date,
                                          pvcalculation = True,
                                          peakpower=peakPower,
                                          optimal_surface_tilt=True, 
                                          optimalangles=True))


            if sum(data[-1][0]['temp_air'])==0:
                raise ValueError("Location has no weather data, trying different location" + "...\n")  
                   
        except ValueError as ve:
            print(ve) 
    
        except:
            print('Location over sea, please provide coordinates on land')

        # Additional locations

        for i in range(locations - 1):
            
            print('Gathering data from additional location ' + str(i+1) + '...\n')
            
            # The distance from the base location, transforming latitude and longitude to kilometers
            lat_dif = (1/110.574) * km_radius
            lat = latitude + lin_loc_df['sin_loc'][i]*lat_dif
            
            # Longitude is based on the actual latitude 
            long_dif = 1/(111.32*abs(math.cos(math.radians(lat)))) * km_radius
            long = longitude + lin_loc_df['cos_loc'][i]*long_dif
            
            # Gaussian randomisation
            lat_dif_gaus = (1/110.574) * gaus_radius
            long_dif_gaus = 1/(111.32*abs(math.cos(math.radians(lat)))) * gaus_radius
            
            mean = [long, lat]
            cov = [[long_dif_gaus,0],
                   [0,lat_dif_gaus]]
            
            x, y = np.random.multivariate_normal(mean, cov, 1).T
            long = x[0]
            lat = y[0]
            
            # Check if location is on land 
            ## If yes: append to the list
            
            long_list = np.linspace(longitude, long, precision)
            lat_list = np.linspace(latitude, lat, precision)
                
            for i in range(0,(precision-1)):
                try: 
                    long_new = long_list[-(i+1)]
                    lat_new = lat_list[-(i+1)]
                    data.append(pvlib_helpers.get_pvgis_hourly(latitude=lat_new,
                                                longitude=long_new, 
                                                start = start_date,
                                                pvcalculation = True,
                                                peakpower=peakPower,
                                                optimal_surface_tilt=True, 
                                                optimalangles=True))
                        
                    if sum(data[-1][0]['temp_air'])==0:
                        del(data[-1])
                        raise ValueError("Location has no weather data, trying different location" + "...\n")  
                    else:
                        break
                            
                except ValueError as ve:
                    print(ve)
                        
                except:
                    print('Location over sea, trying different location' + '...\n')
        
        return data
    
    def dataTransforming(
        dataList,
        covariates = ['poa_direct','poa_sky_diffuse','poa_ground_diffuse','solar_elevation', 'temp_air']):
        
        trainList = []
        covTrainList = []
        testList = []
        covTestList = []
              
        TRAINTESTSPLIT = 0.85
        
        for i in range(len(dataList)):
            
            source_data = dataList[i][0].astype(np.float32)
        
            #Create a TimeSeries object of the target variable
            target_series = TimeSeries.from_series(source_data['P'])
        
            # Create training and validation sets of the target variable
            train, test = target_series.split_after(TRAINTESTSPLIT)
            transformer = Scaler()
            train = transformer.fit_transform(train)
            test = transformer.transform(test)     

            #Create a TimeSeries object of the target variable
            covariate_series = TimeSeries.from_series(source_data[covariates])

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
    
    def model(
        trainList, 
        covTrainList, 
        testList, 
        covTestList,
        modelname,      
        input_length = 24,
        output_length = 24,
        gpu_available = False
        ):
                
        print('Creating model \n')
        
        my_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.00001,
            mode='min',
        )
        
        if gpu_available == False:
            trainer_kwargs = {
                "callbacks": [my_stopper]
            }
        else:
            trainer_kwargs={
                "accelerator": "gpu",
                "gpus": [0],
                "callbacks": [my_stopper]
            }
    
        my_model = BlockRNNModel(
            input_chunk_length=input_length,
            output_chunk_length=output_length,
            model="LSTM",
            hidden_dim=300,
            n_rnn_layers=4,
            dropout=0.4,
            batch_size=32,
            n_epochs=100,
            optimizer_kwargs={"lr": 1e-4},
            model_name= modelname,
            random_state=28,
            save_checkpoints=True,
            force_reset=True,
            pl_trainer_kwargs=trainer_kwargs
        )
    
        print('Training model (this can take a while)\n')
    
        my_model.fit(
            trainList,
            past_covariates=covTrainList,
            val_series=testList,
            val_past_covariates=covTestList,
            verbose=True
        )
    
        my_model = BlockRNNModel.load_from_checkpoint(modelname, best=True)
    
        return my_model
     
    def modelEvaluation():
        return None