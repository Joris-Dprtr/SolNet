from matplotlib import pyplot as plt
import formulas as fm
import scipy.stats as stat
import pandas as pd

def metrics(actual, forecast):
    mse = fm.mse(actual,forecast)
    var_actual = fm.var(actual)
    var_forecast = fm.var(forecast)
    correl = stat.pearsonr(actual,forecast)[0]
    bias = fm.unconditional_bias(actual,forecast)
    conditional_bias_1 = fm.conditional_bias_1(actual,forecast)
    resolution = fm.resolution(actual,forecast)
    conditional_bias_2 = fm.conditional_bias_2(actual,forecast)
    discrimination = fm.discrimination(actual,forecast)

    dict = {'MSE':mse,'Var(x)':var_actual,
                            'Var(y)':var_forecast,
                            'Corr':correl,
                            'Bias':bias,
                            'Conditional bias 1':conditional_bias_1,
                            'Resolution':resolution,
                            'Conditional bias 2':conditional_bias_2,
                            'Discrimination':discrimination}

    metrics = pd.DataFrame(dict,index=['Metrics'])

    return metrics
