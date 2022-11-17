import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import math
import numpy as np

from scipy.interpolate import interp1d

import pvlib_helpers

class Data():

    #Class attributes
    locations = 25          #Number of neighbouring locations to include in the model
    km_radius = 50          #The radius around the actual location to find additional locations
    gaus_radius = 0.5       #The covariance for gaussian noise in km on the radius
    precision = 40          #The closeness to trailing the coastline when locations are close to water

    def __init__(
        self,
        latitude,
        longitude,
        peakPower):



        pass