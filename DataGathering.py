import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import math
import numpy as np

from scipy.interpolate import interp1d

import pvlib_helpers

class Data():

    #Class attributes
    locations = 5          #Number of neighbouring locations to include in the model
    km_radius = 50          #The radius around the actual location to find additional locations
    gaus_radius = 0.5       #The covariance for gaussian noise in km on the radius
    precision = 40          #The closeness to trailing the coastline when locations are close to water

    def __init__(
        self,
        latitude,
        longitude,
        peakPower):

        self.latitude = latitude
        self.longitude = longitude
        self.peakPower = peakPower

    def dataGathering(self):
        data = []
        
        # Make a dataframe for the locations
        lin_loc = np.linspace(0,self.locations,self.locations)
        lin_loc_df = pd.DataFrame({"Locations":lin_loc})

        # Make a sin and cosin pattern based on the number of locations
        lin_loc_df["loc_norm"] = 2 * math.pi * lin_loc_df["Locations"] / lin_loc_df["Locations"].max()

        lin_loc_df["sin_loc"] = np.sin(lin_loc_df["loc_norm"])
        lin_loc_df["cos_loc"] = np.cos(lin_loc_df["loc_norm"])

        # Base location

        try:
            data.append(pvlib_helpers.get_pvgis_hourly(latitude=self.latitude,
                                          longitude=self.longitude, 
                                          optimal_surface_tilt=True, 
                                          optimalangles=True))


            if sum(data[-1][0]['temp_air'])==0:
                raise ValueError("Location has no weather data, trying different location" + "...\n")  
                   
        except ValueError as ve:
            print(ve) 
    
        except:
            print('Location over sea, please provide coordinates on land')

        # Additional locations

        for i in range(self.locations - 1):
            
            print('Additional location ' + str(i+1) + '...\n')
            
            # The distance from the base location, transforming latitude and longitude to kilometers
            lat_dif = (1/110.574) * self.km_radius
            lat = self.latitude + lin_loc_df['sin_loc'][i]*lat_dif
            
            # Longitude is based on the actual latitude 
            long_dif = 1/(111.32*abs(math.cos(math.radians(lat)))) * self.km_radius
            long = self.longitude + lin_loc_df['cos_loc'][i]*long_dif
            
            # Gaussian randomisation
            lat_dif_gaus = (1/110.574) * self.gaus_radius
            long_dif_gaus = 1/(111.32*abs(math.cos(math.radians(lat)))) * self.gaus_radius
            
            mean = [long, lat]
            cov = [[long_dif_gaus,0],
                   [0,lat_dif_gaus]]
            
            x, y = np.random.multivariate_normal(mean, cov, 1).T
            long = x[0]
            lat = y[0]
            
            # Check if location is on land 
            ## If yes: append to the list
            
            long_list = np.linspace(self.longitude, long, self.precision)
            lat_list = np.linspace(self.latitude, lat, self.precision)
                
            for i in range(0,(self.precision-1)):
                try: 
                    long_new = long_list[-(i+1)]
                    lat_new = lat_list[-(i+1)]
                    data.append(pvlib_helpers.get_pvgis_hourly(latitude=lat_new,
                                                longitude=long_new, 
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