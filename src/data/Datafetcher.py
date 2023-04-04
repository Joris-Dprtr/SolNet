import pickle
import datetime
import numpy as np
from util import formulas as fm
from util import pvlib_helpers

class Datafetcher:

    def __init__(
        self,
        latitude,
        longitude,
        peakPower,
        locations = 5,
        start_date = 2005,
        km_radius = 50,                 #The radius around the actual location to find additional locations
        gaus_radius = 0.5,              #The covariance for gaussian noise in km on the radius
        precision = 40):

        self.km_radius = km_radius
        self.gaus_radius = gaus_radius
        self.precision = precision
        
        self.dataset = self._dataGathering(latitude, longitude, peakPower, locations=locations, start_date=start_date)
        
    def _dataGathering(self, latitude, longitude, peakPower, locations=5, start_date=2005):
        
        data = []

        additional_locations = fm.circle_pattern(locations-1)

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
                raise ValueError("Location has no weather data, trying different location" + "...")  
                   
        except ValueError as ve:
            print(ve) 
    
        except:
            print('Location over sea, please provide coordinates on land')

        # Additional locations

        lat_dif = fm.kmToLat(self.km_radius) # Can be calculated once because it does not depend on the current location
        lat_dif_gaus = fm.kmToLat(self.gaus_radius) # The radius around a point to randomise, but in lattitude instead of km

        for i in range(locations - 1):
            
            print(f'Gathering data from additional location {i+1}...')
            
            # The distance from the base location, transforming latitude and longitude to kilometers
            lat_additional = latitude + additional_locations['Sine'][i]*lat_dif
            
            # Longitude is based on the actual latitude and has to be calculated in the loop
            long_dif = fm.kmToLong(self.km_radius, lat_additional)
            long_additional = longitude + additional_locations['Cosine'][i]*long_dif
            
            # Gaussian randomisation longitude (has to be calculated in the loop) 
            long_dif_gaus = fm.kmToLong(self.gaus_radius,lat_additional)
            
            mean = [long_additional, lat_additional]
            cov = [[long_dif_gaus,0], [0,lat_dif_gaus]]
            
            x, y = np.random.multivariate_normal(mean, cov, 1).T
            long_additional, lat_additional = x[0], y[0]
            
            # Check if location is on land 
            ## If yes: append to the list

            long_arr = longitude + (long_additional - longitude) / self.precision * np.arange(self.precision)
            lat_arr = latitude + (lat_additional - latitude) / self.precision * np.arange(self.precision)
            
            long_list = long_arr.tolist()
            lat_list = lat_arr.tolist()
            
            for i in range(0,(self.precision-1)):
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
                        raise ValueError("Location has no weather data, trying different location" + "...")  
                    else:
                        break
                            
                except ValueError as ve:
                    print(ve)
                        
                except:
                    print('Location over sea, trying different location' + '...')
        
        pv_dataset_list = []

        for i in range(len(data)):
            data[i][0].index = data[i][0].index.tz_localize(None).floor('H') 
            pv_dataset_list.append(data[i][0])
        
        return pv_dataset_list
    
    def save_data(self):
        
        now = datetime.datetime.now()
        date_string = now.strftime("%y%m%d_%H%M")
        filename = f"dataset_{date_string}.pkl"
        with open('../data/' + filename, "wb") as f:
            pickle.dump(self.dataset, f)     
        
        return filename