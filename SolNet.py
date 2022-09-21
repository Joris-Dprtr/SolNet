# Do we need to reference PVLIB somehow when we use this? Or make our own version of the API connection?
import pvlib_helpers
import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler

from pytorch_lightning.callbacks import EarlyStopping

class SolNet:
    """
    This class includes the model generation based on a destination provided by the user.
    """
    
    def sourceModel(latitude, longitude, modelname, peakPower = 2.5):
        
        print('Fetching Source Model data\n')
        
        source_data = pvlib_helpers.get_pvgis_hourly(latitude=latitude, 
                                                     longitude=longitude,
                                                     pvcalculation=True,
                                                     peakpower=peakPower,
                                                     optimal_surface_tilt=True, 
                                                     optimalangles=True)
        
        print('Data gathered\n')
        
        print('Transforming data: Removing unused variables, scaling, featurisation \n')
        
        source_data = source_data[0].astype(np.float32)
        
        #Create a TimeSeries object of the target variable
        target_series = TimeSeries.from_series(source_data['P'])
        
        # Create training and validation sets of the target variable
        train, test = target_series.split_after(0.85)
        transformer = Scaler()
        train = transformer.fit_transform(train)
        test = transformer.transform(test)
                
              
            
        #Create a TimeSeries object of the target variable
        covariate_series = TimeSeries.from_series(source_data[['Gb(i)','Gd(i)','Gr(i)','H_sun', 'T2m']])
        
        hour_series = datetime_attribute_timeseries(
            pd.date_range(start=covariate_series.start_time(), freq=covariate_series.freq_str, periods=len(source_data)),
            attribute="hour",
            cyclic=True
            ).astype(np.float32)
    
        covariate_series = covariate_series.stack(hour_series)
        
        
        # Create training and validation sets of the target variable
        cov_train, cov_test = covariate_series.split_after(0.85)
        transformer_2 = Scaler()
        cov_train = transformer_2.fit_transform(cov_train)
        cov_test = transformer_2.transform(cov_test)                
        
        print('Creating model \n')
        

        my_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.00001,
            mode='min',
        )
    
        my_model = BlockRNNModel(
            input_chunk_length=24,
            output_chunk_length=24,
            model="LSTM",
            hidden_size=300,
            n_rnn_layers=4,
            dropout=0.4,
            batch_size=32,
            n_epochs=100,
            optimizer_kwargs={"lr": 1e-4},
            model_name= modelname,
            random_state=28,
            save_checkpoints=True,
            force_reset=True,
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "gpus": [0],
                "callbacks": [my_stopper]
            }
        )
    
        print('Training model\n')
    
        my_model.fit(
            train,
            past_covariates=cov_train,
            val_series=test,
            val_past_covariates=cov_test,
            verbose=True
        )
    
        my_model = BlockRNNModel.load_from_checkpoint(modelname, best=True)
    
        return my_model
        