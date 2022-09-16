# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:06:16 2022

@author: joris
"""

from matplotlib import pyplot as plt
from darts.metrics import rmse, mae, r2_score


def eval_model(series, model, start, past_covariates=None, future_covariates=None, to_retrain=False):
    
    backtest = model.historical_forecasts(series=series, 
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=start, 
                                          retrain=to_retrain,
                                          verbose=True, 
                                          forecast_horizon=24)
    
    series[-len(backtest)-336:].plot()
    backtest.plot(label='Backtest')
    print('Backtest MAE = {}'.format(mae(series, backtest)))
    print('Backtest R2 = {}'.format(r2_score(series, backtest)))
    print('Backtest RMSE = {}'.format(rmse(series, backtest)) + '\n')
    plt.show() 
    
    error = series[-len(backtest):].pd_series() - backtest.pd_series()
    error.rolling(49, center=True).mean().plot()
    plt.show()
    
    return backtest