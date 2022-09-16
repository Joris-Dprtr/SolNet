# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:20:15 2022

@author: joris
"""

from darts.models import BlockRNNModel, NBEATSModel
from darts import TimeSeries
from darts.metrics import rmse, mae, r2_score

from matplotlib import pyplot as plt

import pandas as pd

def LSTM_output(modelName, train_transformed, cov_train, test_transformed, cov_test):
    
    
    from pytorch_lightning.callbacks import EarlyStopping

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
        model_name= modelName,
        random_state=28,
        save_checkpoints=True,
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "gpus": [0],
            "callbacks": [my_stopper]
            }
        )
    
    print('training model')
    
    my_model.fit(
        train_transformed,
        past_covariates=cov_train,
        val_series=test_transformed,
        val_past_covariates=cov_test,
        verbose=True
        )
    
    my_model = BlockRNNModel.load_from_checkpoint(modelName, best=True)
    
    return my_model

def NBEATS_output(modelName, train_transformed, cov_train, test_transformed, cov_test):
    
    
    from pytorch_lightning.callbacks import EarlyStopping

    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.00001,
        mode='min',
        )
    
    my_model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=24,
        num_stacks=4,
        num_blocks=4,
        num_layers=2,
        layer_widths=512,
        #expansion_coefficient_dim=5,
        dropout=0.3,
        batch_size=256,
        n_epochs=100,
        optimizer_kwargs={"lr": 1e-4},
        model_name=modelName,
        random_state=12,
        save_checkpoints=True,
        force_reset=True,
        pl_trainer_kwargs={
            "accelerator": "gpu",
            "gpus": [0],
            "callbacks": [my_stopper]
            }
        )
    
    print('training model')
    
    my_model.fit(
        train_transformed,
        past_covariates=cov_train,
        val_series=test_transformed,
        val_past_covariates=cov_test,
        verbose=True
        )
    
    my_model = NBEATSModel.load_from_checkpoint(modelName, best=True)
    
    return my_model


def model_eval(my_model, series_transformed, covariates, start):
    
    print('testing model')
    
    backtest = my_model.historical_forecasts(series=series_transformed,
                                        past_covariates=covariates,
                                        start=pd.to_datetime(start),
                                        retrain=False,
                                        forecast_horizon=24,
                                        stride=24,
                                        last_points_only=False)
    
    df = pd.DataFrame()

    for i in range(len(backtest)):
        test = backtest[i].pd_dataframe()
        df=pd.concat([df,test])
    
    backtest = TimeSeries.from_dataframe(df)
    
    series_transformed[-len(backtest):].plot()
    backtest.plot(label="Backtest")
    print('Backtest RMSE= {}'.format(rmse(series_transformed, backtest)))
    print('Backtest MAE= {}'.format(mae(series_transformed, backtest)))
    print('Backtest R2= {}'.format(r2_score(series_transformed, backtest)))
    plt.show()
    
    series_transformed_df = series_transformed.pd_dataframe()
    series_transformed_df = series_transformed_df.loc[df[:1].index[0]:df[-1:].index[0]]

    series_transformed_slice = TimeSeries.from_dataframe(series_transformed_df)

    errorLSTM = series_transformed_slice.pd_series() - backtest.pd_series()
    errorLSTM_abs = abs(errorLSTM)
    errorLSTM.rolling(24, center=True).mean().plot()
    plt.show()

    # Throw everything together in a dataframe for plots later on

    series_transformed_df = series_transformed_df.rename(columns = {'P':'actual_output'})
    errorLSTM_df = pd.DataFrame(errorLSTM_abs)
    errorLSTM_df = errorLSTM_df.rename(columns = {0:'error'})
    backtest_df = backtest.pd_dataframe()
    backtest_df = backtest_df.rename(columns = {'P':'predicted_output'})

    comparison = pd.concat([series_transformed_df, errorLSTM_df, backtest_df], axis=1)
    
    return comparison

def model_eval_R2(my_model, series_transformed, covariates, start):
    
    print('testing model')
    
    backtest = my_model.historical_forecasts(series=series_transformed,
                                        past_covariates=covariates,
                                        start=pd.to_datetime(start),
                                        retrain=False,
                                        forecast_horizon=24,
                                        stride=24,
                                        last_points_only=False)
    
    df = pd.DataFrame()

    for i in range(len(backtest)):
        test = backtest[i].pd_dataframe()
        df=pd.concat([df,test])
    
    backtest = TimeSeries.from_dataframe(df)
    
    series_transformed[-len(backtest):].plot()
    backtest.plot(label="Backtest")
    print('Backtest RMSE= {}'.format(rmse(series_transformed, backtest)))
    print('Backtest MAE= {}'.format(mae(series_transformed, backtest)))
    print('Backtest R2= {}'.format(r2_score(series_transformed, backtest)))
    plt.show()
    
    R2 = r2_score(series_transformed, backtest)
    
    series_transformed_df = series_transformed.pd_dataframe()
    series_transformed_df = series_transformed_df.loc[df[:1].index[0]:df[-1:].index[0]]

    series_transformed_slice = TimeSeries.from_dataframe(series_transformed_df)

    errorLSTM = series_transformed_slice.pd_series() - backtest.pd_series()
    errorLSTM_abs = abs(errorLSTM)
    errorLSTM.rolling(24, center=True).mean().plot()
    plt.show()

    # Throw everything together in a dataframe for plots later on

    series_transformed_df = series_transformed_df.rename(columns = {'P':'actual_output'})
    errorLSTM_df = pd.DataFrame(errorLSTM_abs)
    errorLSTM_df = errorLSTM_df.rename(columns = {0:'error'})
    backtest_df = backtest.pd_dataframe()
    backtest_df = backtest_df.rename(columns = {'P':'predicted_output'})

    comparison = pd.concat([series_transformed_df, errorLSTM_df, backtest_df], axis=1)
    
    return comparison, R2
    
    