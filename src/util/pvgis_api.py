import requests
import io
import json

import pandas as pd


class PVgis():
    
    def __init__(self,
                 latitude,
                 longitude,
                 start,
                 tilt,
                 azimuth,
                 peakpower):
    
        self.URL = 'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc'
        LOSS = 14        
        
        self.params = {'lat': latitude, 'lon': longitude,
            'startyear':start,
            'outputformat':'json',
            'angle': tilt, 'aspect': azimuth,
            'pvcalculation': 1,
            'components': 1,
            'peakpower': peakpower,
            'loss':LOSS}
        
    def get_pvgis_hourly(self):
        
        data_request = requests.get(self.URL,params=self.params,timeout=120)
        
        if not data_request.ok:
            try:
                err_msg = data_request.json()
            except Exception:
                data_request.raise_for_status()
            else:
                raise requests.HTTPError(err_msg['message'])
            
        filename = io.StringIO(data_request.text)
        try:
            src = json.load(filename)
        except AttributeError:  # str/path has no .read() attribute
            with open(str(filename), 'r') as fbuf:
                src = json.load(fbuf)
                
        #inputs = src['inputs']
        #metadata = src['meta']
        data = pd.DataFrame(src['outputs']['hourly'])
        data.index = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', utc=True)
        data = data.drop('time', axis=1)
        data = data.astype(dtype={'Int': 'int'})
        
        return data