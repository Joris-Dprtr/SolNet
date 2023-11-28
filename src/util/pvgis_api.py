import requests
import io
import json

import pandas as pd


class PVgis:

    def __init__(self,
                 latitude,
                 longitude,
                 start,
                 tilt,
                 azimuth,
                 peak_power,
                 end=None,
                 optimalangles=0):
        """
        API to access PVGIS data
        :param latitude: latitude of the location of interest
        :param longitude: longitude of the location of interest
        :param start: the start date to gather data (minimum = 2005)
        :param tilt: the tilt of the solar panels
        :param azimuth: the direction of the solar panels
        :param peak_power: the peak power of the installation
        """
        self.URL = 'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc'
        LOSS = 14

        self.params = {'lat': latitude, 'lon': longitude,
                       'startyear': start,
                       'endyear': end,
                       'outputformat': 'json',
                       'angle': tilt, 'aspect': azimuth,
                       'optimalangles': optimalangles,
                       'pvcalculation': 1,
                       'components': 1,
                       'peakpower': peak_power,
                       'loss': LOSS,
                       'localtime': 0}

    def get_pvgis_hourly(self):
        """
        Fetch the data from PVGIS
        :return: a dataframe holding the hourly PVGIS data
        """
        data_request = requests.get(self.URL, params=self.params, timeout=120)

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

        data = pd.DataFrame(src['outputs']['hourly'])
        data.index = pd.to_datetime(data['time'], format='%Y%m%d:%H%M', utc=True)
        data = data.drop('time', axis=1)
        data = data.astype(dtype={'Int': 'int'})

        return data
