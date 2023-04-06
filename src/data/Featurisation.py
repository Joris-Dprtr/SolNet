import pandas as pd
import numpy as np
import pickle

class Featurisation():
    
    def __init__(self, data):
        if data is None:
            raise ValueError("Data cannot be None. Please provide a (list of) pandas dataframe(s) or a file path.")
        elif isinstance(data, list):
            self.data = data
        elif isinstance(data, str):
            self.data = self._load_data(data)
        else:
            raise ValueError("Invalid data type provided. Must be a (list of) pandas dataframe(s) or a file path.")
        
        
    def _load_data(self, file_path):
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print("Error: {} doesn't exist.".format(file_path))
        
        return data
    
    def base_features(self, featurelist):
  
        for i in range(len(self.data)):
            self.data[i] = self.data[i][featurelist]
        
        return self.data
    
    def cyclic_features(self, yearly = True, hourly = True):
        
        for i in range(len(self.data)):        
            if(hourly==True):
                self.data[i]['hour_sin'] = np.sin(2*np.pi*self.data[i].index.hour/24)
                self.data[i]['hour_cos'] = np.cos(2*np.pi*self.data[i].index.hour/24)
            if(yearly==True):
                self.data[i]['month_sin'] = np.sin(2*np.pi*self.data[i].index.month/12)
                self.data[i]['month_cos'] = np.cos(2*np.pi*self.data[i].index.month/12)
                
        return self.data
    
