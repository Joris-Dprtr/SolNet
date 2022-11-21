# Do we need to reference PVLIB somehow when we use this? Or make our own version of the API connection?
import pvlib_helpers
import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from DataGathering import SourceData as Source
from DataTransforming import RawData

from pytorch_lightning.callbacks import EarlyStopping

class SolNet():
    def __init__(
        self,
        latitude, 
        longitude, 
        modelname, 
        peakPower,
        gpu_available = False
        ):
        
        print('Fetching Source Model data\n')
        
        source_data = Source.dataGathering(latitude, longitude, peakPower)

        print('Data gathered\n')
        
        print('Transforming data: Removing unused variables, scaling, featurisation \n')
        
        trainList, covTrainList, testList, covTestList = RawData.dataTransforming(source_data)
                
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
            input_chunk_length=24,
            output_chunk_length=24,
            model="LSTM",
            hidden_dim=300,
            n_rnn_layers=4,
            dropout=0.4,
            batch_size=32,
            n_epochs=1,
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