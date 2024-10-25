import os
import numpy as np
import pandas as pd
import xarray as xr
import ocf_blosc2

from datetime import datetime


def _list_folders_in_directory(directory):
    folder_list = []

    # Iterate through the entries in the directory
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)

        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            folder_list.append(entry)

    return folder_list


def _find_date_gaps(folders, date_format="%Y%m%d", suffix="_99.zarr"):
    # Extract dates from folder names and sort them
    dates = sorted(
        datetime.strptime(folder.split('_')[0], date_format)
        for folder in folders if folder.endswith(suffix)
    )

    gaps = []

    # Check for gaps between consecutive dates
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        if gap > 1:
            gaps.append((dates[i - 1].strftime(date_format), dates[i].strftime(date_format), gap - 1))

    return gaps


class Weather_data:

    def __init__(self,
                 path: str,
                 latitude: float,
                 longitude: float,
                 variables: list):
        self.path = path
        self.latitude = latitude
        self.longitude = longitude
        self.variables = variables

    def get_weather_data(self):

        time_averaged = ['aswdifd_s', 'aswdir_s']

        day_folders = _list_folders_in_directory(self.path)
        gaps = _find_date_gaps(day_folders)
        gap_counter = 0

        weather_data = pd.DataFrame()

        for i in range(len(day_folders)):
            dwd_folder = os.path.join(self.path, day_folders[i])
            data = xr.open_zarr(dwd_folder)
            data = data.sel(latitude=self.latitude, longitude=self.longitude, method='nearest')
            variable_data = []

            if gap_counter < len(gaps) and day_folders[i][0:8] == gaps[gap_counter][0]:

                index = data.step[0:24 + 24 * gaps[gap_counter][2]].values
                start = pd.Timestamp(day_folders[i][0:8])

                correct_index = start + index

                for var in self.variables:

                    current_var = data[var][0:24 + 24 * gaps[gap_counter][2]].values

                    if var in time_averaged:
                        instant_dwd = np.zeros(current_var.shape)
                        instant_dwd[0] = 0
                        for j in range(1, len(instant_dwd)):
                            instant_dwd[j] = max(current_var[j] * (j + 1) - np.sum(instant_dwd[:j]),0)
                        current_var = instant_dwd

                    variable_data.append(current_var)

                stacked_data = np.column_stack(variable_data)

                day = pd.DataFrame(data=stacked_data,
                                   index=correct_index,
                                   columns=self.variables)
                gap_counter += 1

            else:

                for var in self.variables:

                    current_var = data[var][0:24].values

                    if var in time_averaged:
                        instant_dwd = np.zeros(current_var.shape)
                        instant_dwd[0] = 0
                        for j in range(1, len(instant_dwd)):
                            instant_dwd[j] = max(current_var[j] * (j + 1) - np.sum(instant_dwd[:j]),0)
                        current_var = instant_dwd

                    variable_data.append(current_var)

                stacked_data = np.column_stack(variable_data)

                index = data.step[0:24].values
                start = pd.Timestamp(day_folders[i][0:8])
                correct_index = start + index

                day = pd.DataFrame(data=stacked_data,
                                   index=correct_index,
                                   columns=self.variables)
            if i == 0:
                weather_data = day
            else:
                weather_data = pd.concat([weather_data, day])

        weather_data = weather_data[~weather_data.index.duplicated(keep='last')]
        complete_index = pd.date_range(start=weather_data.index.min(), end=weather_data.index.max(), freq='H')
        weather_data = weather_data.reindex(complete_index)
        weather_data = weather_data.ffill()

        return weather_data
