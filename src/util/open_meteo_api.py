import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry


class Open_meteo:

    def __init__(self,
                 latitude,
                 longitude,
                 variables,
                 start_date,
                 end_date):

        self.url = "https://archive-api.open-meteo.com/v1/archive"
        self.params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "weather_code", "cloud_cover", "direct_radiation",
                       "diffuse_radiation"],
            "models": "best_match"
            }
        self.variables = ['date'] + variables

    def get_open_meteo_hourly(self):

        # Set up the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        open_meteo = openmeteo_requests.Client(session=retry_session)

        responses = open_meteo.weather_api(self.url, params=self.params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_weather_code = hourly.Variables(2).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
        hourly_direct_radiation = hourly.Variables(4).ValuesAsNumpy()
        hourly_diffuse_radiation = hourly.Variables(5).ValuesAsNumpy()
        date = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left")

        hourly_data = {"date": date,
                       "temperature_2m": hourly_temperature_2m,
                       "relative_humidity_2m": hourly_relative_humidity_2m,
                       "weather_code": hourly_weather_code,
                       "cloud_cover": hourly_cloud_cover,
                       "direct_radiation": hourly_direct_radiation,
                       "diffuse_radiation": hourly_diffuse_radiation}

        hourly_dataframe = pd.DataFrame(data=hourly_data)
        hourly_dataframe = hourly_dataframe[self.variables]

        return hourly_dataframe
